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
        description=(
            "Diffie-Hellman prime for OAuth — Base64 DER body or hex string. "
            "Use IBKR_OAUTH_DH_PRIME_FP to point to the PEM file instead."
        ),
    )
    ibkr_oauth_dh_prime_fp: str = Field(
        default="",
        validation_alias="IBKR_OAUTH_DH_PRIME_FP",
        description="Path to the dhparam.pem file (alternative to IBKR_OAUTH_DH_PRIME).",
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

    def get_oauth_dh_prime_hex(self) -> str:
        """
        Return the DH prime as a hex string, as required by ibind.

        Accepts three input formats (checked in order):
        1. IBKR_OAUTH_DH_PRIME_FP — path to a dhparam.pem file
        2. IBKR_OAUTH_DH_PRIME    — Base64-encoded DER (PEM body without headers)
        3. IBKR_OAUTH_DH_PRIME    — plain hex string (passed through unchanged)

        Returns "" if neither setting is configured.
        """
        # File path takes precedence over inline value
        fp = self.ibkr_oauth_dh_prime_fp.strip()
        if fp:
            try:
                with open(fp) as f:
                    pem = f.read()
                # Strip PEM armour — keep only the Base64 body lines
                lines = [
                    line.strip()
                    for line in pem.splitlines()
                    if line.strip() and not line.startswith("-----")
                ]
                value = "".join(lines)
            except OSError as e:
                raise ValueError(
                    f"Cannot read IBKR_OAUTH_DH_PRIME_FP ({fp!r}): {e}"
                ) from e
        else:
            value = self.ibkr_oauth_dh_prime.strip()

        if not value:
            return ""
        # Already a hex string?
        if all(c in "0123456789abcdefABCDEF" for c in value):
            return value
        # Assume Base64-encoded DER DH parameters block.
        # Structure: SEQUENCE { INTEGER p (prime), INTEGER g (generator) }
        import base64

        try:
            der = base64.b64decode(value + "=" * (-len(value) % 4))
        except Exception:
            return value  # Can't decode; pass through and let ibind report the error

        try:
            pos = 0
            if der[pos] != 0x30:  # Not a SEQUENCE
                return value
            pos += 1
            # Skip SEQUENCE length (definite long or short form)
            if der[pos] & 0x80:
                n = der[pos] & 0x7F
                pos += 1 + n  # skip length-of-length byte(s)
            else:
                pos += 1
            # First value must be an INTEGER (the prime)
            if der[pos] != 0x02:
                return value
            pos += 1
            # Parse INTEGER length
            if der[pos] & 0x80:
                n = der[pos] & 0x7F
                int_len = int.from_bytes(der[pos + 1 : pos + 1 + n], "big")
                pos += 1 + n
            else:
                int_len = der[pos]
                pos += 1
            # Strip leading 0x00 padding byte (ASN.1 uses it to signal positive)
            prime_bytes = der[pos : pos + int_len]
            if prime_bytes and prime_bytes[0] == 0x00:
                prime_bytes = prime_bytes[1:]
            return prime_bytes.hex()
        except Exception:
            return value  # Malformed DER; pass through

    def is_configured(self) -> bool:
        """Check if minimum IBKR credentials are set."""
        return bool(
            self.ibkr_account_id
            and self.get_oauth_consumer_key()
            and self.ibkr_oauth_encryption_key_fp
            and self.ibkr_oauth_signature_key_fp
            and (self.ibkr_oauth_dh_prime or self.ibkr_oauth_dh_prime_fp)
        )


ibkr_config = IbkrSettings()
