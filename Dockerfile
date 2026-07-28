# ==============================================================================
# MULTI-STAGE DOCKERFILE FOR DJANGO MEDICAL CHATBOT (PRODUCTION)
# ==============================================================================

# STAGE 1: Builder Stage - Install dependencies and prepare environment
FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# STAGE 2: Runtime Stage - Final lightweight production image
FROM python:3.10-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application source code
COPY . /app

# Collect Django static files for production
RUN python manage.py collectstatic --noinput

# Expose server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Start Django application via Gunicorn WSGI server
CMD ["gunicorn", "medicalbot_project.wsgi:application", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120"]
