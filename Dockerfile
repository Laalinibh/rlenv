FROM python:3.11-slim

WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create non-root user for HF Spaces (runs as uid 1000)
RUN useradd -m -u 1000 appuser

COPY . /app/env

RUN pip install --no-cache-dir -e .

# Ensure appuser owns the working directory
RUN chown -R appuser:appuser /app/env

ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

USER appuser

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
