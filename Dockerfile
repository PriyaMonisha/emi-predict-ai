# filename: Dockerfile
# purpose:  Multi-stage image for FastAPI serving and MLflow server (Section 11)
# version:  1.0

# ── Stage 1: install dependencies ─────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: runtime image ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (infrastructure.md requirement)
RUN useradd -m -u 1001 appuser

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY . .
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
