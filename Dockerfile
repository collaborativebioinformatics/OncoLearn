# Multi-stage build for OncoLearn
FROM ubuntu:22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    curl \
    wget \
    gnupg \
    lsb-release \
    software-properties-common \
    # Python development headers (required for rpy2)
    python3-dev \
    python3-pip \
    # R dependencies
    r-base \
    r-base-dev \
    # Additional R libraries needed for rpy2
    libreadline-dev \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    libicu-dev \
    # Additional libraries
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    # NBIA Data Retriever dependencies
    libasound2 \
    libgif7 \
    openjdk-11-jre \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install NBIA Data Retriever
RUN wget -q https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4.3/nbia-data-retriever_4.4.3-1_amd64.deb -O /tmp/nbia-data-retriever.deb \
    && dpkg -i /tmp/nbia-data-retriever.deb || true \
    && rm /tmp/nbia-data-retriever.deb \
    # Create symlink for CLI access (executable is in bin subdirectory)
    && ln -sf /opt/nbia-data-retriever/bin/nbia-data-retriever /usr/local/bin/nbia-data-retriever

# Install LLVM 20 from official repository
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y llvm-20 llvm-20-dev \
    && rm -rf /var/lib/apt/lists/*

# Set LLVM environment variables
ENV LLVM_CONFIG=/usr/bin/llvm-config-20
ENV CMAKE_PREFIX_PATH=/usr/lib/llvm-20

# Install uv (Python package manager) system-wide
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Set uv cache directory to avoid hardlink issues
ENV UV_CACHE_DIR=/tmp/uv-cache

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./
COPY src/ ./src/

# Install Python dependencies including dev tools and fusion extra
RUN uv sync

# Copy R environment files
COPY renv.lock ./
COPY renv/ ./renv/
COPY .Rprofile ./

# Install R dependencies
RUN R -e "if (!requireNamespace('renv', quietly = TRUE)) install.packages('renv', repos='https://cloud.r-project.org/')" \
    && R -e "renv::restore()" \
    && R -e "if (!requireNamespace('languageserver', quietly = TRUE)) install.packages('languageserver', repos='https://cloud.r-project.org/')"

# Copy the rest of the project
COPY . .

# Keep container running for VSCode Dev Containers
# VSCode's Jupyter extension uses ipykernel directly (already in dev dependencies)
CMD ["sleep", "infinity"]
