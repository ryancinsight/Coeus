# Multi-stage Docker build for Coeus Deep Learning Framework
# Supports both development and production deployments

# =============================================================================
# Base stage with Rust toolchain and dependencies
# =============================================================================
FROM rust:1.75-slim AS base

# Install system dependencies for GPU support and development
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./

# Copy all crate manifests for dependency resolution
COPY autograd/Cargo.toml ./autograd/
COPY backend/Cargo.toml ./backend/
COPY distributed/Cargo.toml ./distributed/
COPY dtype/Cargo.toml ./dtype/
COPY examples/Cargo.toml ./examples/
COPY hub/Cargo.toml ./hub/
COPY jit/Cargo.toml ./jit/
COPY nn/Cargo.toml ./nn/
COPY optim/Cargo.toml ./optim/
COPY profiling/Cargo.toml ./profiling/
COPY pycoeus/Cargo.toml ./pycoeus/
COPY storage/Cargo.toml ./storage/
COPY tensor/Cargo.toml ./tensor/
COPY tokenizer/Cargo.toml ./tokenizer/
COPY utils/Cargo.toml ./utils/

# Create dummy source files for dependency resolution
RUN mkdir -p \
    autograd/src backend/src distributed/src dtype/src examples/src \
    hub/src jit/src nn/src optim/src profiling/src pycoeus/src \
    storage/src tensor/src tokenizer/src utils/src \
    && echo "fn main() {}" > examples/src/main.rs \
    && for crate in autograd backend distributed dtype hub jit nn optim profiling pycoeus storage tensor tokenizer utils; do \
        echo "fn dummy() {}" > ${crate}/src/lib.rs; \
    done

# Cache dependencies
RUN cargo check --workspace && rm -rf target/debug/deps/*examples*

# =============================================================================
# Development stage with full development environment
# =============================================================================
FROM base AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for testing and examples
RUN pip3 install --break-system-packages \
    torch \
    torchvision \
    numpy \
    matplotlib \
    jupyter \
    pytest \
    black \
    mypy

# Copy source code
COPY . .

# Build with development optimizations
RUN cargo build --workspace

# Set environment variables for development
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV CARGO_INCREMENTAL=1

# Expose ports for development services
EXPOSE 8000 8888

# Default command for development
CMD ["cargo", "test", "--workspace"]

# =============================================================================
# GPU-enabled development stage
# =============================================================================
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS gpu-development

# Install Rust and system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY autograd/Cargo.toml ./autograd/
COPY backend/Cargo.toml ./backend/
COPY distributed/Cargo.toml ./distributed/
COPY dtype/Cargo.toml ./dtype/
COPY examples/Cargo.toml ./examples/
COPY hub/Cargo.toml ./hub/
COPY jit/Cargo.toml ./jit/
COPY nn/Cargo.toml ./nn/
COPY optim/Cargo.toml ./optim/
COPY profiling/Cargo.toml ./profiling/
COPY pycoeus/Cargo.toml ./pycoeus/
COPY storage/Cargo.toml ./storage/
COPY tensor/Cargo.toml ./tensor/
COPY tokenizer/Cargo.toml ./tokenizer/
COPY utils/Cargo.toml ./utils/

# Cache dependencies
RUN mkdir -p autograd/src backend/src && \
    echo "fn main() {}" > examples/src/main.rs && \
    for crate in autograd backend distributed dtype hub jit nn optim profiling pycoeus storage tensor tokenizer utils; do \
        mkdir -p ${crate}/src && \
        echo "fn dummy() {}" > ${crate}/src/lib.rs; \
    done

RUN cargo check --workspace

# Copy source code and build
COPY . .
RUN cargo build --workspace --features gpu

# Set GPU environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV RUST_LOG=info

CMD ["cargo", "test", "--workspace", "--features", "gpu"]

# =============================================================================
# Production build stage
# =============================================================================
FROM base AS builder

# Copy source code
COPY . .

# Build optimized release binaries
RUN cargo build --release --workspace

# Strip debug symbols for smaller binary size
RUN strip target/release/coeus_tensor && \
    strip target/release/coeus_nn && \
    strip target/release/coeus_backend

# =============================================================================
# Minimal production runtime
# =============================================================================
FROM debian:bookworm-slim AS production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r coeus && useradd -r -g coeus coeus

# Set working directory
WORKDIR /app

# Copy built binaries from builder stage
COPY --from=builder /app/target/release/coeus_tensor /usr/local/bin/
COPY --from=builder /app/target/release/coeus_nn /usr/local/bin/
COPY --from=builder /app/target/release/coeus_backend /usr/local/bin/

# Copy examples and documentation
COPY --from=builder /app/examples /app/examples
COPY --from=builder /app/docs /app/docs
COPY --from=builder /app/README.md /app/

# Create directories for model storage and logs
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R coeus:coeus /app

# Switch to non-root user
USER coeus

# Set environment variables for production
ENV RUST_LOG=warn
ENV RUST_BACKTRACE=0
ENV MODEL_PATH=/app/models
ENV LOG_PATH=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "/usr/local/bin/coeus_tensor", "--version" ] || exit 1

# Default command
CMD ["/usr/local/bin/coeus_tensor"]

# =============================================================================
# Python bindings production stage
# =============================================================================
FROM python:3.11-slim AS python-production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r coeus && useradd -r -g coeus coeus

WORKDIR /app

# Copy Python package and install
COPY pycoeus/ /app/
RUN pip install --no-cache-dir -e .

# Copy examples and documentation
COPY examples/ /app/examples/
COPY docs/ /app/docs/
COPY README.md /app/

# Create directories
RUN mkdir -p /app/models /app/logs && \
    chown -R coeus:coeus /app

USER coeus

# Expose port for potential web services
EXPOSE 8000

# Health check for Python bindings
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import coeus; print('Coeus Python bindings healthy')" || exit 1

CMD ["python", "-c", "import coeus; print('Coeus Python bindings ready')"]

# =============================================================================
# CI/CD testing stage
# =============================================================================
FROM base AS ci

# Install additional testing dependencies
RUN apt-get update && apt-get install -y \
    valgrind \
    gdb \
    lldb \
    && rm -rf /var/lib/apt/lists/*

# Install additional Rust tools for CI
RUN rustup component add clippy rustfmt
RUN cargo install cargo-audit cargo-tarpaulin cargo-udeps

# Copy source code
COPY . .

# Set environment for CI testing
ENV RUST_BACKTRACE=1
ENV RUST_LOG=debug
ENV CARGO_INCREMENTAL=0

# Run comprehensive CI pipeline
RUN cargo check --workspace && \
    cargo clippy --workspace -- -D warnings && \
    cargo fmt --check && \
    cargo test --workspace && \
    cargo audit && \
    cargo udeps --workspace

CMD ["cargo", "test", "--workspace", "--doc"]

# =============================================================================
# Documentation generation stage
# =============================================================================
FROM base AS docs

# Install additional documentation tools
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install mdBook for documentation
RUN cargo install mdbook mdbook-mermaid

# Copy source code
COPY . .

# Generate comprehensive documentation
RUN cargo doc --workspace --no-deps && \
    mdbook build docs/ && \
    cargo doc --workspace --document-private-items

# Create documentation archive
RUN tar -czf /docs.tar.gz -C target/doc . && \
    tar -czf /mdbook-docs.tar.gz -C docs/book .

# =============================================================================
# Enterprise deployment stage with Kubernetes
# =============================================================================
FROM production AS enterprise

# Install additional enterprise tools
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy enterprise configuration and scripts
COPY deployment/enterprise/ /app/deployment/
COPY scripts/ /app/scripts/

# Create enterprise directories
RUN mkdir -p /app/config /app/secrets /app/monitoring && \
    chown -R coeus:coeus /app

# Set enterprise environment variables
ENV ENTERPRISE_MODE=true
ENV MONITORING_ENABLED=true
ENV SECURITY_AUDIT=true

# Enterprise health check with additional validation
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD /app/scripts/health-check.sh || exit 1

# Default enterprise command
CMD ["/app/scripts/start-enterprise.sh"]

# =============================================================================
# Default target (production)
# =============================================================================
FROM production
