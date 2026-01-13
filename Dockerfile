# Multi-stage build for OncoLearn
FROM buildpack-deps:jammy AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set uv (Python package manager) environment variables
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv

# Set LLVM environment variables
ENV LLVM_CONFIG=/usr/bin/llvm-config-20
ENV CMAKE_PREFIX_PATH=/usr/lib/llvm-20

# Build argument for GPU-specific PyTorch extras
ARG GPU_EXTRA=cu130

# Install system dependencies
# Note: buildpack-deps already includes: build-essential, git, curl, wget, etc.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    # Python development headers (required for rpy2)
    python3-dev \
    python3-pip \
    # R dependencies
    r-base \
    r-base-dev \
    # Additional compatability libraries
    libreadline-dev \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    libicu-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install NBIA Data Retriever
RUN mkdir -p /usr/share/desktop-directories \
    && wget -q https://github.com/CBIIT/NBIA-TCIA/releases/download/DR-4_4_3-TCIA-20240916-1/nbia-data-retriever_4.4.3-1_amd64.deb -O /tmp/nbia-data-retriever.deb \
    && dpkg -i /tmp/nbia-data-retriever.deb || true \
    && apt --fix-broken install -y \
    && ln -sf /opt/nbia-data-retriever/bin/nbia-data-retriever /usr/local/bin/nbia-data-retriever \
    && rm /tmp/nbia-data-retriever.deb \
    && rm -rf /var/lib/apt/lists/*

# Install PDC CLI
RUN mkdir -p /opt/pdc-client \
    && wget https://pdc-download-clients.s3.amazonaws.com/pdc-client_v1.0.8_Ubuntu_x64.zip -O /tmp/pdc-client.zip \
    && unzip /tmp/pdc-client.zip -d /tmp \
    && unzip /tmp/pdc-client_v1.0.8_Ubuntu_x64/pdc-client_CL_v1.0.8_Ubuntu_x64.zip -d /opt/pdc-client \
    && chmod +x /opt/pdc-client/pdc-client \
    && ln -sf /opt/pdc-client/pdc-client /usr/local/bin/pdc_client \
    && rm -rf /tmp/pdc-client.zip /tmp/pdc-client_v1.0.8_Ubuntu_x64 /tmp/__MACOSX

# Install LLVM 20 from official repository
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends llvm-20 llvm-20-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager) system-wide
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Set working directory
WORKDIR /workspace

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install Python dependencies with BuildKit cache mount for much faster WSL builds
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --extra ${GPU_EXTRA}

# Copy R dependency files
COPY .Rprofile renv.lock ./
COPY renv/ ./renv/

# Install R dependencies
RUN R -e "if (!requireNamespace('renv', quietly = TRUE)) install.packages('renv', repos='https://cloud.r-project.org/')" \
    && R -e "renv::restore()" \
    && R -e "if (!requireNamespace('languageserver', quietly = TRUE)) install.packages('languageserver', repos='https://cloud.r-project.org/')"

# Copy source code (now that dependencies are cached)
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY notebooks/ ./notebooks/
COPY README.md ./

# Keep container running for VSCode Dev Containers
# VSCode's Jupyter extension uses ipykernel directly (already in dev dependencies)
CMD ["sleep", "infinity"]
