FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (for layer caching)
COPY pyproject.toml .

# Install dependencies with uv (no venv inside container — just install to system)
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the app
COPY . .

# Create directories that need to persist (can be overridden by volumes)
RUN mkdir -p /app/data /app/chroma_db

# Expose Flask port
EXPOSE 5000

# Entrypoint
CMD ["python", "app.py"]
