# AGI-in-a-Box: Geometry-Aware Router Service
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Builder - Install dependencies and build wheels
# =============================================================================
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1
RUN pip install poetry==${POETRY_VERSION}

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies (production only)
RUN poetry install --no-root --only main 2>/dev/null || poetry install --no-root

# Install optional topology dependencies for enhanced routing
RUN poetry run pip install numpy scipy || true
RUN poetry run pip install ripser persim || true

# Copy application code
COPY . .

# Install the project itself
RUN poetry install --only main 2>/dev/null || poetry install

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/geometry_router /app/geometry_router
COPY --from=builder /app/pyproject.toml /app/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Router configuration defaults
ENV ROUTER_HOST="0.0.0.0"
ENV ROUTER_PORT="8080"
ENV ROUTER_GRPC_PORT="50051"
ENV LOG_LEVEL="INFO"

# Model backend configuration
ENV OPENAI_API_BASE="http://ollama:11434/v1"
ENV OPENAI_MODEL_NAME="mixtral"

# Topology configuration
ENV TOPOLOGY_APPROXIMATE="true"
ENV SINKHORN_ITERATIONS="20"
ENV MAX_RECURSION_DEPTH="5"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${ROUTER_PORT}/health || exit 1

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8080 50051

# Default command - can be overridden
CMD ["python", "-m", "geometry_router.server"]
