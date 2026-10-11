"""The one answer to "which ``.env`` file may this process read?".

Every ``BaseSettings`` subclass in ``src/`` takes its ``env_file`` from
:func:`default_settings_env_file`, and ``src.config._parse_env_file`` (the reader
behind ``get_env_value``) honours it too. Under pytest the answer is ``None``, so
no code path ingests the operator's gitignored ``.env``. It lives in its own
module because ``src.config`` is heavy and the IBKR settings classes must not
import it. Guarded by ``tests/config/test_env_file_policy.py``.
"""

import os

ENV_FILE_NAME = ".env"
DISABLE_DOTENV_VARIABLE = "PYTEST_DISABLE_DOTENV"


def default_settings_env_file() -> str | None:
    """``.env``, or None when the test harness forbids reading the operator's."""
    if os.environ.get(DISABLE_DOTENV_VARIABLE) == "true":
        return None
    return ENV_FILE_NAME
