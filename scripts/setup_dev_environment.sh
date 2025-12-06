#!/bin/bash

# Coeus Development Environment Setup Script
# This script sets up a complete development environment for Coeus contributors

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     OS=linux;;
        Darwin*)    OS=macos;;
        CYGWIN*|MINGW*|MSYS*) OS=windows;;
        *)          OS=unknown;;
    esac
    log_info "Detected OS: $OS"
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."

    # Check available memory (minimum 4GB)
    if command_exists free; then
        MEM_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
        if [ "$MEM_GB" -lt 4 ]; then
            log_warning "System has ${MEM_GB}GB RAM. 4GB+ recommended for development."
        else
            log_success "Memory check passed: ${MEM_GB}GB available"
        fi
    fi

    # Check available disk space (minimum 10GB)
    if command_exists df; then
        DISK_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
        if [ "$DISK_GB" -lt 10 ]; then
            log_warning "Low disk space: ${DISK_GB}GB available. 10GB+ recommended."
        else
            log_success "Disk space check passed: ${DISK_GB}GB available"
        fi
    fi
}

# Install Rust toolchain
install_rust() {
    log_info "Installing Rust toolchain..."

    if command_exists rustc; then
        RUST_VERSION=$(rustc --version | cut -d' ' -f2)
        log_info "Rust already installed: $RUST_VERSION"
    else
        log_info "Installing Rust via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
        log_success "Rust installed successfully"
    fi

    # Install required components
    log_info "Installing Rust components..."
    rustup component add clippy rustfmt miri
    log_success "Rust components installed"
}

# Install development tools
install_dev_tools() {
    log_info "Installing development tools..."

    # Install cargo tools
    CARGO_TOOLS=(
        "cargo-tarpaulin"    # Code coverage
        "cargo-criterion"    # Benchmarking
        "cargo-udeps"        # Unused dependencies
        "cargo-audit"        # Security auditing
        "cargo-edit"         # Cargo.toml editing
        "cargo-workspaces"   # Workspace management
        "cargo-release"      # Release management
        "mdbook"             # Documentation
        "cargo-make"         # Task runner
    )

    for tool in "${CARGO_TOOLS[@]}"; do
        if ! cargo install --list | grep -q "^$tool "; then
            log_info "Installing $tool..."
            cargo install "$tool"
        else
            log_info "$tool already installed"
        fi
    done

    log_success "Development tools installed"
}

# Setup Python environment (for PyCoeus)
setup_python() {
    log_info "Setting up Python environment..."

    if ! command_exists python3 && ! command_exists python; then
        log_warning "Python not found. Please install Python 3.8+ manually for PyCoeus development."
        return
    fi

    # Use python3 if available, otherwise python
    PYTHON_CMD="python3"
    if ! command_exists python3 && command_exists python; then
        PYTHON_CMD="python"
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $PYTHON_VERSION"

    # Install pip if not available
    if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        log_info "Installing pip..."
        curl -sS https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD
    fi

    # Install Python development dependencies
    log_info "Installing Python development dependencies..."
    $PYTHON_CMD -m pip install --user --upgrade pip
    $PYTHON_CMD -m pip install --user maturin pytest numpy torch

    log_success "Python environment setup complete"
}

# Setup IDE configuration
setup_ide_config() {
    log_info "Setting up IDE configuration..."

    # Create .vscode directory if it doesn't exist
    mkdir -p .vscode

    # Create VS Code settings if they don't exist
    if [ ! -f .vscode/settings.json ]; then
        cat > .vscode/settings.json << 'EOF'
{
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.cargo.features": ["all"],
    "rust-analyzer.procMacro.enable": true,
    "rust-analyzer.diagnostics.disabled": [
        "inactive-code"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll": true,
        "source.organizeImports": true
    },
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer",
        "editor.formatOnSave": true
    },
    "[toml]": {
        "editor.formatOnSave": true
    },
    "[markdown]": {
        "editor.formatOnSave": true
    },
    "files.associations": {
        "Cargo.toml": "toml",
        "*.rs": "rust"
    },
    "git.autofetch": true,
    "git.enableSmartCommit": true,
    "terminal.integrated.shell.windows": "C:\\Program Files\\Git\\bin\\bash.exe",
    "terminal.integrated.shell.linux": "/bin/bash",
    "terminal.integrated.shell.osx": "/bin/bash"
}
EOF
        log_success "VS Code settings created"
    fi

    # Create VS Code launch configuration
    if [ ! -f .vscode/launch.json ]; then
        cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Basic Usage Example",
            "type": "lldb",
            "request": "launch",
            "cargo": {
                "args": [
                    "build",
                    "--example=basic_usage"
                ]
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Debug Neural Network Example",
            "type": "lldb",
            "request": "launch",
            "cargo": {
                "args": [
                    "build",
                    "--example=neural_network"
                ]
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Debug Unit Tests",
            "type": "lldb",
            "request": "launch",
            "cargo": {
                "args": [
                    "test",
                    "--lib",
                    "--package=coeus-tensor"
                ]
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF
        log_success "VS Code launch configuration created"
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    # Check Rust
    if command_exists rustc; then
        RUSTC_VERSION=$(rustc --version)
        log_success "Rust: $RUSTC_VERSION"
    else
        log_error "Rust not found"
        return 1
    fi

    # Check Cargo
    if command_exists cargo; then
        CARGO_VERSION=$(cargo --version)
        log_success "Cargo: $CARGO_VERSION"
    else
        log_error "Cargo not found"
        return 1
    fi

    # Check development tools
    DEV_TOOLS=("cargo-tarpaulin" "cargo-criterion" "cargo-audit")
    for tool in "${DEV_TOOLS[@]}"; do
        if command_exists "$tool"; then
            log_success "$tool: installed"
        else
            log_warning "$tool: not found"
        fi
    done

    # Test basic compilation
    log_info "Testing basic compilation..."
    if cargo check --quiet; then
        log_success "Basic compilation check passed"
    else
        log_error "Compilation check failed"
        return 1
    fi

    log_success "Installation verification complete!"
}

# Setup Git hooks (optional)
setup_git_hooks() {
    log_info "Setting up Git hooks..."

    # Create pre-commit hook
    mkdir -p .git/hooks
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

echo "Running pre-commit checks..."

# Run cargo fmt check
echo "Checking code formatting..."
if ! cargo fmt --check; then
    echo "❌ Code formatting check failed. Run 'cargo fmt' to fix."
    exit 1
fi

# Run clippy
echo "Running clippy..."
if ! cargo clippy -- -D warnings; then
    echo "❌ Clippy check failed."
    exit 1
fi

# Run tests
echo "Running tests..."
if ! cargo test --quiet; then
    echo "❌ Tests failed."
    exit 1
fi

echo "✅ All pre-commit checks passed!"
EOF

    chmod +x .git/hooks/pre-commit
    log_success "Git hooks configured"
}

# Print next steps
print_next_steps() {
    cat << 'EOF'

🎉 Development environment setup complete!

Next steps:
1. Review CONTRIBUTING.md for contribution guidelines
2. Run 'cargo build' to build the project
3. Run 'cargo test' to run tests
4. Run 'cargo run --example basic_usage' to try an example
5. Check docs/ for documentation

Useful commands:
• cargo build          - Build the project
• cargo test           - Run all tests
• cargo run --example  - Run an example
• cargo doc --open     - Build and open documentation
• cargo fmt            - Format code
• cargo clippy         - Run linter

For help:
• Check CONTRIBUTING.md
• Open an issue on GitHub
• Join our community Discord (coming soon)

Happy coding! 🚀

EOF
}

# Main setup function
main() {
    echo "🚀 Setting up Coeus development environment..."
    echo

    detect_os
    check_system_requirements

    echo
    install_rust
    echo
    install_dev_tools
    echo
    setup_python
    echo
    setup_ide_config
    echo
    setup_git_hooks
    echo
    verify_installation

    if [ $? -eq 0 ]; then
        print_next_steps
    else
        log_error "Setup completed with errors. Please check the output above."
        exit 1
    fi
}

# Run main function
main "$@"
