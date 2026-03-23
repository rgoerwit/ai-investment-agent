# ═══════════════════════════════════════════════════════════════════════════
# Multi-Agent Investment Analysis System - Modern Dockerfile (2025)
# ═══════════════════════════════════════════════════════════════════════════
#
# IMPROVEMENTS OVER LEGACY VERSION:
# - Multi-stage build (reduces image size by ~40%)
# - Poetry 2.x with modern dependency groups
# - ChromaDB system dependencies included
# - Non-root user for security
# - Proper health check without HTTP dependency
# - Build-time argument for Python version flexibility
# ═══════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder - Install dependencies in isolated environment
# ────────────────────────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

ARG POETRY_VERSION=2.1.1

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry using official installer
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /build

# Copy dependency manifests and README (required by pyproject.toml)
COPY pyproject.toml poetry.lock* README.md ./

# Configure Poetry to not create virtual env (we're already in a container)
ENV POETRY_VIRTUALENVS_CREATE=false

# Install dependencies to system Python (builder stage)
# Updated: --only main replaces deprecated --no-dev
# Note: We use --no-root to skip installing the project itself in builder stage
# The actual project code is copied in the runtime stage
RUN poetry install --only main --no-root --no-interaction --no-ansi

# ────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime - Minimal production image
# ────────────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# Install only runtime dependencies (SQLite for ChromaDB + bash for scripts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (don't run as root in production)
RUN groupadd -r agent && useradd -r -g agent -u 1000 agent

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy application code
COPY --chown=agent:agent src/ ./src/
COPY --chown=agent:agent prompts/ ./prompts/
COPY --chown=agent:agent scripts/ ./scripts/

# Normalise source-file permissions: COPY preserves host mode bits, so
# directories that are mode 700 on the host would be unreadable by any user
# other than the owner. a+rX makes all source/prompts/scripts files readable
# and directories traversable regardless of what the host had.
RUN chmod -R a+rX /app/src /app/prompts /app/scripts

# Create directories for bind-mounted local persistence
RUN mkdir -p /app/chroma_db /app/results /app/data_cache /app/images /app/scratch \
    && chown -R agent:agent /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app \
    INVESTMENT_AGENT_CONTAINER=1 \
    CHROMA_PERSIST_DIRECTORY=/app/chroma_db \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    RESULTS_DIR=/app/results \
    DATA_CACHE_DIR=/app/data_cache \
    IMAGES_DIR=/app/images \
    ANONYMIZED_TELEMETRY=False

# Switch to non-root user
USER agent

# Health check - verify Python and the lightweight config module import cleanly
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import src.config" || exit 1

# Entrypoint uses module syntax for clean imports
ENTRYPOINT ["python", "-m", "src.main"]

# Default arguments (override at runtime)
CMD ["--ticker", "AAPL", "--quick"]
