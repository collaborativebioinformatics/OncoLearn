# Multi-stage build for OncoLearn

# Build argument for GPU-specific PyTorch extras (must be before FROM to be global)
ARG GPU_EXTRA=cpu

# ============================================================================
# Stage 1: LLVM Builder
# ============================================================================
FROM buildpack-deps:jammy-curl AS llvm-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends llvm-20 llvm-20-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 2: NBIA Data Retriever Builder
# ============================================================================
FROM buildpack-deps:jammy-curl AS nbia-builder

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies that NBIA needs
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libgtk-3-0 \
    libasound2 \
    libgif7 \
    libice6 \
    libsm6 \
    xdg-utils

# Install NBIA Data Retriever
RUN mkdir -p /usr/share/desktop-directories /usr/share/applications \
    && wget -q https://github.com/CBIIT/NBIA-TCIA/releases/download/DR-4_4_3-TCIA-20240916-1/nbia-data-retriever_4.4.3-1_amd64.deb -O /tmp/nbia-data-retriever.deb \
    && dpkg -i /tmp/nbia-data-retriever.deb \
    && rm /tmp/nbia-data-retriever.deb

# ============================================================================
# Stage 3: PDC CLI Builder
# ============================================================================
FROM buildpack-deps:jammy-curl AS pdc-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/pdc-client \
    && wget https://pdc-download-clients.s3.amazonaws.com/pdc-client_v1.0.8_Ubuntu_x64.zip -O /tmp/pdc-client.zip \
    && unzip /tmp/pdc-client.zip -d /tmp \
    && unzip /tmp/pdc-client_v1.0.8_Ubuntu_x64/pdc-client_CL_v1.0.8_Ubuntu_x64.zip -d /opt/pdc-client \
    && chmod +x /opt/pdc-client/pdc-client \
    && rm -rf /tmp/pdc-client.zip /tmp/pdc-client_v1.0.8_Ubuntu_x64 /tmp/__MACOSX

# ============================================================================
# Stage 4: Python Dependencies Builder
# ============================================================================
FROM python:3.12 AS python-builder

ARG GPU_EXTRA

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_CACHE_DIR=/workspace/.uv-cache
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv
ENV UV_NO_PROGRESS=0
ENV UV_HTTP_TIMEOUT=300
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

# Install uv (build tools already included in python:3.12)
RUN pip install uv

WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install Python dependencies with cache mount
RUN uv sync --verbose --extra ${GPU_EXTRA}

# ============================================================================
# Stage 5: R Dependencies Builder
# ============================================================================
FROM r-base:latest AS r-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV RENV_PATHS_CACHE=/workspace/.renv-cache

# Install build dependencies for R packages with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev

WORKDIR /workspace

# Copy R dependency files
COPY .Rprofile renv.lock ./
COPY renv/ ./renv/

# Install R dependencies with cache mount and parallel compilation
RUN --mount=type=cache,target=/workspace/.renv-cache,sharing=locked \
    R -e "options(Ncpus=parallel::detectCores()); \
          if (!requireNamespace('renv', quietly = TRUE)) install.packages('renv', repos='https://cloud.r-project.org/'); \
          if (!requireNamespace('languageserver', quietly = TRUE)) install.packages('languageserver', repos='https://cloud.r-project.org/'); \
          renv::restore();"

# ============================================================================
# Stage 6: Dev Runtime Image
# ============================================================================
FROM buildpack-deps:jammy AS dev

ARG GPU_EXTRA

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set uv and Python environment variables
ENV UV_CACHE_DIR=/workspace/.uv-cache
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv
ENV UV_NO_PROGRESS=0
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

# Set LLVM environment variables
ENV LLVM_CONFIG=/usr/bin/llvm-config-20
ENV CMAKE_PREFIX_PATH=/usr/lib/llvm-20

# Install minimal runtime system dependencies
# Note: Python comes from the venv, R runtime deps are minimal
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy LLVM runtime from llvm-builder stage (exclude dev files to reduce size)
COPY --from=llvm-builder /usr/lib/llvm-20/lib/*.so* /usr/lib/llvm-20/lib/
COPY --from=llvm-builder /usr/lib/llvm-20/bin/llvm-config /usr/lib/llvm-20/bin/
COPY --from=llvm-builder /usr/bin/llvm-config-20 /usr/bin/

# Copy NBIA Data Retriever from nbia-builder stage
COPY --from=nbia-builder /opt/nbia-data-retriever /opt/nbia-data-retriever
RUN mkdir -p /usr/share/desktop-directories \
    && ln -sf /opt/nbia-data-retriever/bin/nbia-data-retriever /usr/local/bin/nbia-data-retriever

# Copy PDC CLI from pdc-builder stage
COPY --from=pdc-builder /opt/pdc-client /opt/pdc-client
RUN ln -sf /opt/pdc-client/pdc-client /usr/local/bin/pdc_client

# Copy Python virtual environment from python-builder stage
COPY --from=python-builder /workspace/.venv /workspace/.venv

# Copy R installation and libraries from r-builder stage
COPY --from=r-builder /usr/lib/R /usr/lib/R
COPY --from=r-builder /usr/local/lib/R /usr/local/lib/R
COPY --from=r-builder /usr/bin/R /usr/bin/R
COPY --from=r-builder /usr/bin/Rscript /usr/bin/Rscript
COPY --from=r-builder /workspace/renv /workspace/renv

# Copy uv from python-builder (faster than downloading)
COPY --from=python-builder /usr/local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /workspace

# Copy source code and configs
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .Rprofile renv.lock ./
    
# Keep container running for VSCode Dev Containers
# VSCode's Jupyter extension uses ipykernel directly (already in dev dependencies)
CMD ["sleep", "infinity"]

# ============================================================================
# Stage 7: Production PyTorch Runtime (Optimized for Small Size)
# ============================================================================
FROM python:3.12-slim AS pytorch-prod

ARG GPU_EXTRA

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set Python environment variables
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

# Install only essential runtime dependencies for PyTorch
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only Python virtual environment from python-builder stage
COPY --from=python-builder /workspace/.venv /workspace/.venv

# Set working directory
WORKDIR /workspace

# Copy only essential source code (exclude dev tools, notebooks, docs)
COPY src/ ./src/
COPY configs/ ./configs/

# Default command for training/inference
CMD ["python", "-m", "oncolearn.train"]
