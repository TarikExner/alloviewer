FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

ARG TORCH_REQUIREMENT="torch>=2.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata first for better Docker layer caching.
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

# Install CPU-only PyTorch.
RUN python -m pip install --no-cache-dir --upgrade \
        pip \
        setuptools \
        wheel \
    && python -m pip install --no-cache-dir \
        "${TORCH_REQUIREMENT}" \
        --index-url https://download.pytorch.org/whl/cpu

# Copy the full project.
COPY . /app

# Install AlloViewer and verify that PyTorch is CPU-only.
RUN python -m pip install --no-cache-dir . \
    && python -m pip check \
    && python -c "import torch; print('Installed PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); assert torch.version.cuda is None, f'Expected CPU-only PyTorch, but CUDA build is {torch.version.cuda}'"

RUN mkdir -p \
    /data \
    /data/jobs \
    /data/segmented \
    /data/_thumbs \
    /data/plate_layouts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
