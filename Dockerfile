# Multi-stage build for Jitterbug 2.0
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml* .

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY docs/ ./docs/
COPY scripts/ ./scripts/
COPY README.md .
COPY LICENSE* .

# Install the package
RUN uv pip install --system -e .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r jitterbug && useradd -r -g jitterbug jitterbug

# Set working directory
WORKDIR /app

# Copy installed packages and application from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app .

# Create directories for data and output
RUN mkdir -p /app/data /app/output && \
    chown -R jitterbug:jitterbug /app && \
    chmod +x /app/scripts/docker-entrypoint.sh

# Switch to non-root user
USER jitterbug

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["api"]