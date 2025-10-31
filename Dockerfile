# Coeus Multimodal AI Platform - Production Docker Configuration
# Multi-stage build for optimized production containers

# =============================================================================
# Base Stage: Common dependencies and build tools
# =============================================================================
FROM rust:1.70-slim AS base

# Install system dependencies for Rust compilation and runtime
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./

# Copy all crate source files for dependency resolution
COPY autograd/ ./autograd/
COPY backend/ ./backend/
COPY dtype/ ./dtype/
COPY storage/ ./storage/
COPY tensor/ ./tensor/
COPY nn/ ./nn/
COPY optim/ ./optim/
COPY distributed/ ./distributed/
COPY foundation/ ./foundation/
COPY audio/ ./audio/
COPY jit/ ./jit/
COPY profiling/ ./profiling/
COPY pycoeus/ ./pycoeus/
COPY tokenizer/ ./tokenizer/
COPY utils/ ./utils/
COPY hub/ ./hub/
COPY coeus-semantic-api/ ./coeus-semantic-api/
COPY examples/ ./examples/

# Cache dependencies
RUN cargo fetch

# =============================================================================
# Builder Stage: Compile the application
# =============================================================================
FROM base AS builder

# Set build profile for optimized binary
ENV RUSTFLAGS="-C target-cpu=generic -C opt-level=3 -C codegen-units=1"
ENV CARGO_PROFILE_RELEASE_LTO=true
ENV CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
ENV CARGO_PROFILE_RELEASE_PANIC=abort

# Build all binaries in release mode
RUN cargo build --release --workspace

# =============================================================================
# Runtime Stage: Minimal production container
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r coeus && useradd -r -g coeus coeus

# Set working directory
WORKDIR /app

# Copy compiled binaries from builder stage
COPY --from=builder /app/target/release/end_to_end_integration /app/
COPY --from=builder /app/target/release/semantic_api_server /app/
COPY --from=builder /app/target/release/automated_clip_research /app/
COPY --from=builder /app/target/release/enhanced_clip_training /app/
COPY --from=builder /app/target/release/clip_distributed_training /app/
COPY --from=builder /app/target/release/clip_training_pipeline /app/
COPY --from=builder /app/target/release/clip_evaluation /app/
COPY --from=builder /app/target/release/clip_research_integration /app/
COPY --from=builder /app/target/release/clip_semantic_search /app/
COPY --from=builder /app/target/release/benchmark_clip_production /app/
COPY --from=builder /app/target/release/multimodal_transformer_demo /app/

# Create data directories with proper permissions
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R coeus:coeus /app

# Switch to non-root user
USER coeus

# Health check for container
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/app/semantic_api_server", "--health-check"] || exit 1

# Default command - run the integration test suite
CMD ["/app/end_to_end_integration"]

# =============================================================================
# GPU Runtime Stage: For GPU-accelerated workloads
# =============================================================================
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS gpu-runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r coeus && useradd -r -g coeus coeus

WORKDIR /app

# Copy GPU-enabled binaries
COPY --from=builder /app/target/release/end_to_end_integration /app/
COPY --from=builder /app/target/release/semantic_api_server /app/
COPY --from=builder /app/target/release/enhanced_clip_training /app/
COPY --from=builder /app/target/release/clip_distributed_training /app/
COPY --from=builder /app/target/release/clip_semantic_search /app/

# Create data directories
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R coeus:coeus /app

USER coeus

# GPU health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD nvidia-smi || exit 1

CMD ["/app/end_to_end_integration"]

# =============================================================================
# Development Stage: Full development environment
# =============================================================================
FROM base AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    git \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create development user
RUN groupadd -r dev && useradd -r -g dev -s /bin/bash dev && \
    mkdir -p /home/dev && chown dev:dev /home/dev

USER dev
WORKDIR /home/dev

# Mount source code for development
VOLUME ["/app"]

CMD ["bash"]

# =============================================================================
# Labels and metadata
# =============================================================================
LABEL org.opencontainers.image.title="Coeus Multimodal AI Platform"
LABEL org.opencontainers.image.description="Production-ready multimodal AI platform with CLIP, semantic search, and automated research capabilities"
LABEL org.opencontainers.image.version="0.2.0"
LABEL org.opencontainers.image.authors="Ryan Clanton <ryanclanton@protomail.com>"
LABEL org.opencontainers.image.source="https://github.com/ryancinsight/Coeus"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"

# Security labels
LABEL org.opencontainers.image.vendor="Coeus AI"
LABEL org.opencontainers.image.documentation="https://docs.coeus.ai"

# Platform capabilities
LABEL ai.coeus.capabilities="multimodal,clip,semantic-search,gpu-acceleration,distributed-training"
LABEL ai.coeus.models="clip-vit-base,clip-vit-large,transformer-xl,attention-mechanisms"
LABEL ai.coeus.features="hyperparameter-optimization,experiment-tracking,automated-research"