"""
Multi-Agent Trading System - Agent Prompts Registry

The versioned JSON files in the prompts/ directory are the single canonical
prompt source. Loading fails loudly (PromptLoadError) when the directory or a
required agent prompt is missing — there is no inline fallback corpus (the
former one had drifted from the JSON files and was removed June 2026).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.config import config, get_env_value
from src.runtime_config import get_runtime_config

logger = structlog.get_logger(__name__)


class PromptLoadError(RuntimeError):
    """Raised when the canonical JSON prompt corpus is missing or incomplete."""


# Agent prompts the analysis pipeline cannot run without. article_writer and
# editor_in_chief are post-analysis tooling with their own fallbacks and are
# deliberately not required here.
REQUIRED_AGENT_KEYS = frozenset(
    {
        "apac_regional_specialist",
        "bear_researcher",
        "bull_researcher",
        "consultant",
        "foreign_language_analyst",
        "fundamentals_analyst",
        "global_forensic_auditor",
        "junior_fundamentals_analyst",
        "legal_counsel",
        "macro_context_analyst",
        "market_analyst",
        "neutral_analyst",
        "news_analyst",
        "portfolio_manager",
        "research_manager",
        "risky_analyst",
        "safe_analyst",
        "sentiment_analyst",
        "trader",
        "valuation_calculator",
        "value_trap_detector",
    }
)


@dataclass
class AgentPrompt:
    """
    Structured prompt with metadata for version tracking.
    """

    agent_key: str
    agent_name: str
    version: str
    system_message: str
    category: str = "general"
    requires_tools: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "local"
    langfuse_name: str | None = None
    langfuse_label: str | None = None
    langfuse_version: str | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PromptRegistry:
    """Central registry for all agent prompts with version tracking."""

    def __init__(self, prompts_dir: str | None = None):
        # Use explicit path if provided, otherwise fall back to config
        self.prompts_dir = Path(prompts_dir) if prompts_dir else config.prompts_dir
        self.prompts: dict[str, AgentPrompt] = {}
        loaded_count = self._load_json_prompts()
        missing = REQUIRED_AGENT_KEYS - set(self.prompts)
        if missing:
            raise PromptLoadError(
                f"Required prompt files missing or unparseable in "
                f"{self.prompts_dir}: {sorted(missing)}"
            )
        logger.info("prompts_loaded", total=loaded_count, path=str(self.prompts_dir))

    def _load_json_prompts(self) -> int:
        """Load the canonical prompt corpus from JSON files."""
        if not self.prompts_dir.exists():
            raise PromptLoadError(f"Prompts directory missing: {self.prompts_dir}")

        loaded_count = 0
        for json_file in sorted(self.prompts_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                agent_key = data.get("agent_key")
                if not agent_key:
                    logger.warning("json_file_missing_agent_key", file=json_file.name)
                    continue

                prompt = AgentPrompt(**data)
                self.prompts[agent_key] = prompt
                loaded_count += 1
                logger.debug(
                    "prompt_loaded", agent_key=agent_key, version=prompt.version
                )

            except Exception as e:
                from src.error_safety import summarize_exception

                logger.error(
                    "prompt_file_load_failed",
                    file=json_file.name,
                    **summarize_exception(e, operation="prompt_file_load"),
                )
        return loaded_count

    def _langfuse_prompt_enabled(self) -> bool:
        runtime_config = get_runtime_config(config)
        return bool(
            runtime_config.langfuse_enabled
            and config.langfuse_prompt_fetch_enabled
            and config.get_langfuse_public_key()
            and config.get_langfuse_secret_key()
        )

    def _resolve_langfuse_prompt(
        self, agent_key: str, prompt: AgentPrompt | None
    ) -> AgentPrompt | None:
        if prompt is None or not self._langfuse_prompt_enabled():
            return prompt

        try:
            from langfuse import get_client

            client = get_client()
            if not hasattr(client, "get_prompt"):
                raise RuntimeError("Langfuse client does not support prompt fetch")
            prompt_client = client.get_prompt(
                name=agent_key,
                label=config.langfuse_prompt_label,
                type="text",
                cache_ttl_seconds=config.langfuse_prompt_cache_ttl_seconds,
                fallback=prompt.system_message,
            )
            resolved_text = (
                getattr(prompt_client, "prompt", None) or prompt.system_message
            )
            langfuse_version = getattr(prompt_client, "version", None)
            merged_metadata = {
                **prompt.metadata,
                "prompt_source": "langfuse",
                "prompt_name": agent_key,
                "prompt_label": config.langfuse_prompt_label,
                "local_prompt_version": prompt.version,
            }
            return AgentPrompt(
                agent_key=prompt.agent_key,
                agent_name=prompt.agent_name,
                version=prompt.version,
                system_message=resolved_text,
                category=prompt.category,
                requires_tools=prompt.requires_tools,
                metadata=merged_metadata,
                source="langfuse",
                langfuse_name=agent_key,
                langfuse_label=config.langfuse_prompt_label,
                langfuse_version=str(langfuse_version) if langfuse_version else None,
            )
        except Exception as exc:
            from src.error_safety import summarize_exception

            logger.warning(
                "langfuse_prompt_fetch_failed",
                agent_key=agent_key,
                **summarize_exception(exc, operation="langfuse_prompt_fetch"),
            )
            return prompt

    def get(self, agent_key: str) -> AgentPrompt | None:
        """Get prompt by agent key, checking env var override first."""
        env_var = f"PROMPT_{agent_key.upper()}"
        override_message = get_env_value(env_var)
        if override_message:
            base_prompt = self.prompts.get(agent_key)
            if base_prompt:
                prompt = AgentPrompt(
                    agent_key=agent_key,
                    agent_name=base_prompt.agent_name,
                    version=f"{base_prompt.version}-env",
                    system_message=override_message,
                    category=base_prompt.category,
                    requires_tools=base_prompt.requires_tools,
                    metadata={"source": "environment"},
                    source="environment",
                )
                return prompt

        return self._resolve_langfuse_prompt(agent_key, self.prompts.get(agent_key))

    def get_all(self) -> dict[str, AgentPrompt]:
        """Get all registered prompts."""
        return self.prompts.copy()

    def list_keys(self) -> list:
        """List all registered prompt keys."""
        return list(self.prompts.keys())

    def export_to_json(self, output_dir: str | None = None):
        """Export all prompts to JSON files."""
        export_dir = Path(output_dir or self.prompts_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        for agent_key, prompt in self.prompts.items():
            output_file = export_dir / f"{agent_key}.json"

            prompt_dict = {
                "agent_key": prompt.agent_key,
                "agent_name": prompt.agent_name,
                "version": prompt.version,
                "system_message": prompt.system_message,
                "category": prompt.category,
                "requires_tools": prompt.requires_tools,
                "metadata": prompt.metadata,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(prompt_dict, f, indent=2, ensure_ascii=False)

            logger.info("prompt_exported", agent_key=agent_key, file=str(output_file))


# Global registry instance
_registry = None


def get_registry() -> PromptRegistry:
    """Get or create the global prompt registry."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def get_prompt(agent_key: str) -> AgentPrompt | None:
    """Convenience function to get a prompt by key."""
    return get_registry().get(agent_key)


def get_all_prompts() -> dict[str, AgentPrompt]:
    """Convenience function to get all prompts."""
    return get_registry().get_all()


def export_prompts(output_dir: str | None = None):
    """Convenience function to export prompts."""
    get_registry().export_to_json(output_dir)
