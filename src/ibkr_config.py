"""
Configuration for IBKR integration.

Follows src/config.py pattern: Pydantic Settings, SecretStr for credentials, .env loading.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class IbkrSettings(BaseSettings):
    """IBKR connection and portfolio management settings."""

    # --- IBKR OAuth Credentials ---
    ibkr_account_id: str = Field(
        default="",
        validation_alias="IBKR_ACCOUNT_ID",
        description="IBKR account ID",
    )
    ibkr_oauth_consumer_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="IBKR_OAUTH_CONSUMER_KEY",
        description="OAuth consumer key from IBKR",
    )
    ibkr_oauth_access_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="IBKR_OAUTH_ACCESS_TOKEN",
        description="OAuth access token",
    )
    ibkr_oauth_access_token_secret: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="IBKR_OAUTH_ACCESS_TOKEN_SECRET",
        description="OAuth access token secret",
    )
    ibkr_oauth_encryption_key_fp: str = Field(
        default="",
        validation_alias="IBKR_OAUTH_ENCRYPTION_KEY_FP",
        description="Path to OAuth encryption key file",
    )
    ibkr_oauth_signature_key_fp: str = Field(
        default="",
        validation_alias="IBKR_OAUTH_SIGNATURE_KEY_FP",
        description="Path to OAuth signature key file",
    )
    ibkr_oauth_dh_prime: str = Field(
        default="",
        validation_alias="IBKR_OAUTH_DH_PRIME",
        description="Diffie-Hellman prime for OAuth",
    )

    # --- Portfolio Management Defaults ---
    ibkr_max_analysis_age_days: int = Field(
        default=14,
        ge=1,
        validation_alias="IBKR_MAX_ANALYSIS_AGE_DAYS",
        description="Max analysis age in days before considered stale",
    )
    ibkr_drift_threshold_pct: float = Field(
        default=15.0,
        ge=0.0,
        validation_alias="IBKR_DRIFT_THRESHOLD_PCT",
        description="Price drift threshold percentage for staleness",
    )
    ibkr_cash_buffer_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        validation_alias="IBKR_CASH_BUFFER_PCT",
        description="Cash reserve as fraction of portfolio value",
    )
    ibkr_rate_limit_per_sec: int = Field(
        default=10,
        ge=1,
        validation_alias="IBKR_RATE_LIMIT_PER_SEC",
        description="Max IBKR API requests per second",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=False,
        populate_by_name=True,
    )

    def get_oauth_consumer_key(self) -> str:
        return self.ibkr_oauth_consumer_key.get_secret_value()

    def get_oauth_access_token(self) -> str:
        return self.ibkr_oauth_access_token.get_secret_value()

    def get_oauth_access_token_secret(self) -> str:
        return self.ibkr_oauth_access_token_secret.get_secret_value()

    def is_configured(self) -> bool:
        """Check if minimum IBKR credentials are set."""
        return bool(
            self.ibkr_account_id
            and self.get_oauth_consumer_key()
            and self.ibkr_oauth_encryption_key_fp
            and self.ibkr_oauth_signature_key_fp
        )


ibkr_config = IbkrSettings()
