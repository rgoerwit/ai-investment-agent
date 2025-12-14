# Gemini Flash Model Compatibility Guide (December 2025)

## Executive Summary

**Current Situation**: The system currently uses `gemini-3-pro-preview` for ALL agents due to Gemini 2.x Flash model incompatibility with `langchain-google-genai 2.1.12`.

**Root Cause**: API structure issues with `bind_tools()`, NOT context window limitations.

**Impact**: 10 out of 12 agents use expensive Pro model, ~10-25x cost increase vs intended Flash models.

**Long-term Solution**: Upgrade to `langchain-google-genai 4.0.0+` (consolidated SDK).

---

## Critical Finding: Most Agents ALWAYS Use "Quick" Model

Looking at `src/graph.py:262-280`, the system architecture is:

```python
# ALWAYS use quick_think_llm (regardless of --quick flag):
- Market Analyst        (line 262)
- Social Analyst        (line 263)
- News Analyst          (line 264)
- Fundamentals Analyst  (line 265)
- Bull Researcher       (line 266)
- Bear Researcher       (line 267)
- Trader                (line 277)
- Risk Analysts (3x)    (lines 278-280)
# Total: 10 agents

# Conditional (depends on --quick flag):
- Research Manager      (line 271 or 274)
- Portfolio Manager     (line 272 or 275)
# Total: 2 agents

# OpenAI (optional):
- Consultant            (line 283)
```

**THIS MEANS**: With current workaround (`QUICK_MODEL=gemini-3-pro-preview`), **100% of Gemini agents use expensive Pro model in BOTH normal and --quick modes**.

---

## Root Cause Analysis

### NOT a Context Window Issue

Research shows the problem is **NOT** related to context window size limitations. Gemini 2.x Flash models support large contexts (up to 1M tokens).

### API Structure Issue: `bind_tools()` Incompatibility

Per [LangGraph Issue #4780](https://github.com/langchain-ai/langgraph/issues/4780):

**Problem**: Gemini 2.x produces empty message parts during tool calling, violating Google's API requirement: `"contents.parts must not be empty"`

**Trigger**: Using `llm.bind_tools(tools)` method

**Why It Happens**:
1. LangChain calls Gemini with `bind_tools()` to enable function calling
2. Gemini 2.x Flash models generate responses with empty `text` fields during tool invocation
3. LangChain passes these back to Gemini API
4. Google API rejects empty parts → failure/hang

**Evidence**:
- Works: `runnable = prompt | llm`
- Fails: `runnable = prompt | llm.bind_tools(tools)`

### Why Gemini 3 Pro Works

Gemini 3 Pro has more robust tool calling implementation that doesn't generate empty message parts, avoiding the API validation error.

---

## Can the Code Work with Both Cheap and Expensive Models?

**Short Answer**: Not reliably with current SDK (`langchain-google-genai 2.1.12`).

**Long Answer**:

### Option 1: Upgrade SDK (RECOMMENDED)
Upgrade to `langchain-google-genai 4.0.0+` which uses Google's consolidated `google-genai` SDK instead of legacy `google-ai-generativelanguage`.

**Benefits**:
- May fix Gemini 2.x Flash tool calling issues
- Better long-term support
- Unified API for Gemini Developer API and Vertex AI

**Risks**:
- Breaking changes possible (major version bump)
- Requires testing full workflow
- May need code adjustments

**How to Test**:
```bash
# Create test branch
git checkout -b test-langchain-4

# Upgrade SDK
poetry add langchain-google-genai@^4.0.0

# Run tests
poetry run pytest tests/ -v

# Test live analysis
poetry run python -m src.main --ticker AAPL --quick --brief

# If successful, update .env to use Flash models
QUICK_MODEL=gemini-2.0-flash  # Test this specifically
```

### Option 2: Message Sanitization Workaround (COMPLEX)

Per [LangGraph Issue #4780](https://github.com/langchain-ai/langgraph/issues/4780), implement message sanitization to filter empty parts:

```python
def sanitize_ai_message(message):
    """Filter empty text entries from AI message content."""
    if hasattr(message, 'content') and isinstance(message.content, list):
        message.content = [
            part if part.get('text') else {'text': '[No content]', **part}
            for part in message.content
        ]
    return message
```

**Why We Haven't Implemented This**:
1. Requires modifying LangChain agent execution flow
2. Brittle - breaks if LangChain internals change
3. Doesn't address root cause (SDK compatibility)
4. SDK upgrade is cleaner solution

### Option 3: Selective Model Assignment (PARTIAL FIX)

Use Pro for tool-heavy agents, Flash for simple analysis:

```python
# In src/graph.py - Example modification (NOT IMPLEMENTED)
# Simple data gathering (no tool calling) → Flash OK
market_llm = create_flash_model_safe()  # gemini-2.0-flash
news_llm = create_flash_model_safe()

# Complex tool calling → Pro required
fund_llm = create_pro_model()  # gemini-3-pro-preview
bull_llm = create_pro_model()
```

**Problem**: ALL our agents use tools extensively (get_financial_metrics, get_news, etc.), so this doesn't help.

---

## Small Adjustments Without Dumbing Down?

### What WON'T Work

1. **Reducing context size** - Not the issue
2. **Simplifying prompts** - Doesn't affect tool calling
3. **Disabling tools** - Breaks core functionality
4. **Rate limiting adjustments** - Unrelated to failures

### What MIGHT Work

1. **SDK Upgrade** (primary recommendation)
   - Clean solution targeting root cause
   - No code dumbing down required
   - May enable Flash models as intended

2. **Retry Logic Enhancement** (minor help)
   ```python
   # In src/llms.py:81-95
   # ALREADY IMPLEMENTED: max_retries handles transient failures
   # But won't fix systematic tool calling bugs
   ```

3. **Model Selection by Complexity** (limited benefit)
   - Use Pro only for Fundamentals/Bull/Bear (high tool use)
   - Use Flash for News/Social (lighter tool use)
   - **But**: All agents call tools, so savings minimal

---

## Recent Discussions and Documentation

### langchain-google-genai 4.0.0 Release (Dec 8, 2024)

Per [langchain-google releases](https://github.com/langchain-ai/langchain-google/releases):

**Major Changes**:
- Migrated from `google-ai-generativelanguage` to consolidated `google-genai` SDK
- Unified support for Gemini Developer API and Vertex AI
- Deprecations in `langchain-google-vertexai`

**Tool Calling Improvements**: Not explicitly documented, but SDK consolidation suggests better integration.

### Community Workarounds

From [Issue #4780](https://github.com/langchain-ai/langgraph/issues/4780) (May-June 2025):
- Message sanitization confirmed working for some users
- Issue marked "resolved" but users report persistence
- Suggests incomplete fix in 2.x SDK line

### Current Status (Dec 2025)

- **langchain-google-genai 2.1.12**: Has known Flash model tool calling bugs
- **langchain-google-genai 4.0.0+**: Available for 1 year, may fix issues
- **Gemini 3 Pro**: Works reliably but expensive ($2/$12 per 1M tokens)
- **Gemini 2.x Flash**: Broken with 2.1.12, unknown status with 4.0.0+

---

## System Failure vs Data Unavailability

### Current Problem

The system doesn't distinguish between:

1. **System Failures** (internal bugs):
   - LLM tool calling hangs
   - API timeout errors
   - Rate limiting (429 errors)
   - Model compatibility bugs

2. **Data Gaps** (external reality):
   - API doesn't have P/E ratio for this ticker
   - Company doesn't disclose US revenue
   - No analyst coverage data available

### Why This Matters

**System Failures** require user action:
- Fix configuration
- Upgrade dependencies
- Adjust rate limits
- Switch models

**Data Gaps** are expected:
- Small cap stocks have incomplete data
- International stocks lack US data
- New IPOs have limited history
- **NO USER ACTION NEEDED** - system should handle gracefully

### Proposed Solution (NOT YET IMPLEMENTED)

Add failure detection in toolkit and agents:

```python
# Example for src/toolkit.py
class ToolExecutionResult:
    success: bool
    data: Optional[dict]
    failure_type: Literal["system_error", "data_unavailable", "success"]
    error_detail: Optional[str]

def get_financial_metrics(ticker: str) -> ToolExecutionResult:
    try:
        result = fetch_from_apis(ticker)
        if result.empty:
            return ToolExecutionResult(
                success=True,  # Tool worked, just no data
                data=None,
                failure_type="data_unavailable",
                error_detail="No financial data available for ticker"
            )
        return ToolExecutionResult(
            success=True,
            data=result,
            failure_type="success",
            error_detail=None
        )
    except TimeoutError:
        return ToolExecutionResult(
            success=False,
            data=None,
            failure_type="system_error",
            error_detail="API timeout - check network/rate limits"
        )
    except Exception as e:
        return ToolExecutionResult(
            success=False,
            data=None,
            failure_type="system_error",
            error_detail=f"Unexpected error: {str(e)}"
        )
```

Then Portfolio Manager could:
- **Data Gaps**: Mark metrics as N/A, continue analysis
- **System Failures**: Issue WARNING, recommend HOLD, alert user to fix

---

## Recommended Actions

### Immediate (Current State - Dec 2025)

✅ **DONE**: System configured to use `gemini-3-pro-preview` for all agents
- Works reliably
- Expensive but functional
- Documented in README and .env

### Short-term (Next Sprint)

1. **Test SDK Upgrade**:
   ```bash
   # On feature branch
   poetry add langchain-google-genai@^4.0.0
   poetry run pytest tests/ -v
   # Test with gemini-2.0-flash if tests pass
   ```

2. **Implement System Failure Detection**:
   - Update toolkit to distinguish error types
   - Add failure_type to agent state
   - Portfolio Manager checks for system failures before recommending BUY

3. **Update Monitoring**:
   - Log when system failures occur
   - Track failure rates by agent/tool
   - Alert on elevated failure rates

### Long-term (Roadmap)

1. **If SDK 4.0.0+ Works**:
   - Update production to use Flash models
   - Reduce costs by 10-25x
   - Document upgrade process

2. **If SDK 4.0.0+ Fails**:
   - Implement message sanitization workaround
   - Continue using Pro model
   - Wait for SDK fixes from Google/LangChain

3. **Architecture Improvements**:
   - Decouple model selection from agent design
   - Allow per-agent model override via config
   - Support mixed Pro/Flash deployments

---

## Cost Impact Summary

**Current Configuration** (All Pro):
```
Analysis cost: ~$0.05-0.10 per ticker
Batch 300 tickers: ~$15-30
Monthly (3000 analyses): ~$150-300
```

**If Flash Works** (10 agents Flash, 2 Pro):
```
Analysis cost: ~$0.002-0.005 per ticker
Batch 300 tickers: ~$0.60-1.50
Monthly (3000 analyses): ~$6-15
```

**Savings Potential**: 90-95% cost reduction (~$135-285/month for heavy users)

---

## Testing Checklist

Before deploying Flash models in production:

- [ ] Upgrade to langchain-google-genai 4.0.0+
- [ ] Run full test suite (800 tests pass)
- [ ] Test with gemini-2.0-flash on sample tickers:
  - [ ] Large cap (e.g., 9984.T SoftBank) - should succeed
  - [ ] Small cap (e.g., 1681.HK Consun) - should handle gracefully
  - [ ] International (e.g., 2330.TW TSMC) - verify data quality
- [ ] Verify tool calling works (check for DATA_BLOCK generation)
- [ ] Monitor for hangs/timeouts
- [ ] Check cost reduction in LangSmith
- [ ] Run batch analysis (50+ tickers) for stability
- [ ] Document any issues and workarounds

---

**Last Updated**: December 14, 2025
**Current Version**: langchain-google-genai 2.1.12 (legacy SDK)
**Recommended Upgrade**: langchain-google-genai 4.0.0+ (consolidated SDK)
**Status**: Investigation complete, upgrade path identified

## Sources

- [LangGraph Issue #4780 - Gemini 2.5 Fails with LangGraph Agent](https://github.com/langchain-ai/langgraph/issues/4780)
- [langchain-google Issue #638 - Gemini 2.0 flash tool usage not working](https://github.com/langchain-ai/langchain-google/issues/638)
- [langchain-google Releases](https://github.com/langchain-ai/langchain-google/releases)
- [LangChain Google GenAI Package](https://pypi.org/project/langchain-google-genai/)
