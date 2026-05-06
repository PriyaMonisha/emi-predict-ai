# filename: docker/streamlit/Dockerfile
# purpose:  Slim Streamlit UI image — HTTP client only, no ML stack
# version:  1.0

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY docker/streamlit/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 appuser

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY streamlit_app.py .
COPY pages/ pages/
COPY src/__init__.py src/
COPY src/ui/ src/ui/
COPY .streamlit/ .streamlit/
COPY data/processed/sample_batch_template.csv data/processed/

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
