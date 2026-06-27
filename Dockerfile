FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml

# Include these only if they exist in your project.
# If one does not exist, remove that COPY line.
COPY README.md /app/README.md

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

RUN mkdir -p /data /data/jobs /data/segmented /data/_thumbs /data/plate_layouts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
