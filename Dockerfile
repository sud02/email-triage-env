FROM python:3.11-slim

# Metadata for HF Spaces
LABEL maintainer="openenv-submission"
LABEL org.openenv.version="1.0"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
