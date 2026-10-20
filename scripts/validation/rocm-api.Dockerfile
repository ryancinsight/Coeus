FROM rocm/dev-ubuntu-24.04:7.2.4

# SDK/compiler tooling only. The runtime invocation passes no device or GPU.
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates curl git build-essential pkg-config python3 rocm-hip-runtime-dev
ENV RUSTUP_HOME=/opt/rocm-api-rustup CARGO_HOME=/opt/rocm-api-cargo
ENV PATH=/opt/rocm-api-cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain none && \
    rustup toolchain install 1.97.0 --profile minimal && \
    chmod -R a+rX /opt/rocm-api-rustup /opt/rocm-api-cargo
