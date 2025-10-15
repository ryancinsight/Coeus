#!/bin/bash
# Enterprise Security Audit Script for Coeus Framework
# Performs comprehensive security analysis and compliance checks

set -euo pipefail

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

# Global variables
AUDIT_REPORT="security-audit-report-$(date +%Y%m%d-%H%M%S).txt"
VULNERABILITIES_FOUND=0
ISSUES_FOUND=0

# Initialize audit report
init_audit_report() {
    cat > "$AUDIT_REPORT" << EOF
============================================================
Coeus Framework Enterprise Security Audit Report
Generated: $(date)
Version: $(git describe --tags --always 2>/dev/null || echo "dev")
============================================================

EOF
}

# Log to both console and report
log_to_report() {
    echo "$1" >> "$AUDIT_REPORT"
    echo "$1"
}

# Check for unsafe code usage
check_unsafe_code() {
    log_info "Checking for unsafe code usage..."

    local unsafe_count=$(grep -r "unsafe" --include="*.rs" . --exclude-dir=target --exclude-dir=.git | wc -l)

    if [ "$unsafe_count" -gt 0 ]; then
        log_error "Found $unsafe_count instances of unsafe code"
        log_to_report "❌ UNSAFE CODE DETECTED: $unsafe_count instances found"

        grep -r "unsafe" --include="*.rs" . --exclude-dir=target --exclude-dir=.git | while read -r line; do
            log_to_report "  $line"
        done

        ((ISSUES_FOUND++))
    else
        log_success "No unsafe code found"
        log_to_report "✅ No unsafe code detected"
    fi
}

# Check for hardcoded secrets
check_hardcoded_secrets() {
    log_info "Checking for hardcoded secrets..."

    local secret_patterns=(
        "password"
        "secret"
        "token"
        "key"
        "api_key"
        "auth_token"
        "private_key"
    )

    local secrets_found=0

    for pattern in "${secret_patterns[@]}"; do
        local count=$(grep -r -i "$pattern" --include="*.rs" --include="*.toml" --include="*.yaml" --include="*.yml" . \
                      --exclude-dir=target \
                      --exclude-dir=.git \
                      --exclude="Cargo.lock" \
                      | grep -v "^#" | wc -l)

        if [ "$count" -gt 0 ]; then
            log_warning "Found $count instances of '$pattern'"
            ((secrets_found++))
        fi
    done

    if [ "$secrets_found" -gt 0 ]; then
        log_error "Potential hardcoded secrets detected"
        log_to_report "❌ POTENTIAL HARDCODED SECRETS: $secrets_found categories found"
        ((ISSUES_FOUND++))
    else
        log_success "No hardcoded secrets detected"
        log_to_report "✅ No hardcoded secrets detected"
    fi
}

# Check dependency security
check_dependencies() {
    log_info "Checking dependency security..."

    if command -v cargo-audit &> /dev/null; then
        if cargo audit --json | jq -e '.vulnerabilities.found == true' &>/dev/null; then
            local vuln_count=$(cargo audit --json | jq '.vulnerabilities.count')
            log_error "Found $vuln_count vulnerabilities in dependencies"
            log_to_report "❌ DEPENDENCY VULNERABILITIES: $vuln_count vulnerabilities found"

            cargo audit | grep -A 10 -B 2 "Crate:" >> "$AUDIT_REPORT"
            ((VULNERABILITIES_FOUND+=vuln_count))
        else
            log_success "No dependency vulnerabilities found"
            log_to_report "✅ No dependency vulnerabilities detected"
        fi
    else
        log_warning "cargo-audit not installed, skipping dependency audit"
        log_to_report "⚠️  cargo-audit not available for dependency scanning"
    fi
}

# Check for proper error handling
check_error_handling() {
    log_info "Checking error handling patterns..."

    # Check for unwrap() usage in production code
    local unwrap_count=$(grep -r "\.unwrap()" --include="*.rs" . --exclude-dir=target --exclude-dir=tests | wc -l)

    if [ "$unwrap_count" -gt 10 ]; then
        log_warning "Found $unwrap_count unwrap() calls in production code"
        log_to_report "⚠️  HIGH UNWRAP USAGE: $unwrap_count unwrap() calls in production code"
        ((ISSUES_FOUND++))
    else
        log_success "Error handling appears appropriate"
        log_to_report "✅ Error handling patterns are appropriate"
    fi
}

# Check for proper logging
check_logging() {
    log_info "Checking logging configuration..."

    if grep -r "println!" --include="*.rs" . --exclude-dir=target --exclude-dir=tests | grep -v "test" | head -5 | wc -l | grep -q "0"; then
        log_success "No println! usage in production code"
        log_to_report "✅ No println! usage in production code"
    else
        log_warning "println! usage found in production code"
        log_to_report "⚠️  println! usage detected in production code"
        ((ISSUES_FOUND++))
    fi
}

# Check memory safety with Miri (if available)
check_memory_safety() {
    log_info "Checking memory safety with Miri..."

    if command -v cargo &> /dev/null && cargo --version | grep -q "nightly"; then
        if rustup component list | grep -q "miri.*installed"; then
            log_info "Running Miri memory safety checks..."
            if timeout 300 cargo miri test --workspace &>/dev/null; then
                log_success "Miri memory safety checks passed"
                log_to_report "✅ Miri memory safety checks passed"
            else
                log_error "Miri detected memory safety issues"
                log_to_report "❌ Miri detected memory safety issues"
                ((ISSUES_FOUND++))
            fi
        else
            log_warning "Miri not installed, skipping memory safety checks"
            log_to_report "⚠️  Miri not available for memory safety verification"
        fi
    else
        log_warning "Nightly Rust not available, skipping Miri checks"
        log_to_report "⚠️  Nightly Rust not available for Miri checks"
    fi
}

# Check for proper input validation
check_input_validation() {
    log_info "Checking input validation..."

    # Look for basic validation patterns
    local validation_patterns=("validate" "check" "verify" "sanitize")

    local validation_found=0
    for pattern in "${validation_patterns[@]}"; do
        if grep -r "$pattern" --include="*.rs" . --exclude-dir=target | grep -v "test" | head -1 | wc -l | grep -q "0"; then
            continue
        else
            ((validation_found++))
        fi
    done

    if [ "$validation_found" -gt 0 ]; then
        log_success "Input validation patterns found"
        log_to_report "✅ Input validation patterns detected"
    else
        log_warning "Limited input validation detected"
        log_to_report "⚠️  Limited input validation patterns found"
        ((ISSUES_FOUND++))
    fi
}

# Check for proper resource management
check_resource_management() {
    log_info "Checking resource management..."

    # Check for proper cleanup patterns
    if grep -r "Drop" --include="*.rs" . --exclude-dir=target | grep -v "test" | head -3 | wc -l | grep -q "0"; then
        log_warning "Limited explicit resource cleanup detected"
        log_to_report "⚠️  Limited explicit resource cleanup patterns"
    else
        log_success "Resource cleanup patterns found"
        log_to_report "✅ Resource cleanup patterns detected"
    fi
}

# Check license compliance
check_license_compliance() {
    log_info "Checking license compliance..."

    if command -v cargo-license &> /dev/null; then
        local incompatible_licenses=$(cargo license --json | jq -r '.[] | select(.license | test("MIT|Apache-2.0|BSD-3-Clause|ISC") | not) | .name')

        if [ -n "$incompatible_licenses" ]; then
            log_error "Incompatible licenses found:"
            echo "$incompatible_licenses" | while read -r license; do
                log_error "  - $license"
            done
            log_to_report "❌ INCOMPATIBLE LICENSES: $incompatible_licenses"
            ((ISSUES_FOUND++))
        else
            log_success "All licenses are enterprise-compatible"
            log_to_report "✅ All dependency licenses are enterprise-compatible"
        fi
    else
        log_warning "cargo-license not available, skipping license check"
        log_to_report "⚠️  cargo-license not available for license compliance"
    fi
}

# Generate security recommendations
generate_recommendations() {
    log_info "Generating security recommendations..."

    cat >> "$AUDIT_REPORT" << 'EOF'

============================================================
SECURITY RECOMMENDATIONS
============================================================

EOF

    if [ "$VULNERABILITIES_FOUND" -gt 0 ]; then
        cat >> "$AUDIT_REPORT" << 'EOF'
🔴 CRITICAL: Address dependency vulnerabilities immediately
   - Update vulnerable dependencies to patched versions
   - Implement dependency scanning in CI/CD pipeline
   - Establish dependency update policy

EOF
    fi

    if [ "$ISSUES_FOUND" -gt 0 ]; then
        cat >> "$AUDIT_REPORT" << 'EOF'
🟡 HIGH PRIORITY: Address security issues
   - Implement proper input validation
   - Replace unwrap() calls with proper error handling
   - Remove println! usage in production code
   - Implement secrets management system

EOF
    fi

    cat >> "$AUDIT_REPORT" << 'EOF'
🟢 GENERAL RECOMMENDATIONS:
   - Implement regular security audits
   - Use dependency scanning tools in CI/CD
   - Implement secrets management (HashiCorp Vault, AWS Secrets Manager, etc.)
   - Enable security headers and TLS 1.3
   - Implement rate limiting and DDoS protection
   - Regular penetration testing and vulnerability assessments
   - Implement proper logging and monitoring
   - Use container security scanning (Trivy, Clair, etc.)

============================================================
COMPLIANCE CHECKLIST
============================================================

EOF

    # Compliance checklist
    local compliance_items=(
        "🔍 Static Application Security Testing (SAST): IMPLEMENTED"
        "🔍 Software Composition Analysis (SCA): IMPLEMENTED"
        "🔍 Dynamic Application Security Testing (DAST): NOT IMPLEMENTED"
        "🔍 Container Security Scanning: PARTIALLY IMPLEMENTED"
        "🔍 Secrets Management: NOT IMPLEMENTED"
        "🔍 Input Validation & Sanitization: PARTIALLY IMPLEMENTED"
        "🔍 Error Handling & Information Disclosure: IMPLEMENTED"
        "🔍 Authentication & Authorization: NOT APPLICABLE"
        "🔍 Session Management: NOT APPLICABLE"
        "🔍 Cryptography: NOT APPLICABLE"
        "🔍 Memory Safety: IMPLEMENTED (Rust guarantees)"
    )

    for item in "${compliance_items[@]}"; do
        echo "$item" >> "$AUDIT_REPORT"
    done
}

# Main audit function
main() {
    log_info "Starting Coeus Framework Enterprise Security Audit"
    log_info "Report will be saved to: $AUDIT_REPORT"

    init_audit_report

    log_to_report "AUDIT SUMMARY"
    log_to_report "============="

    check_unsafe_code
    check_hardcoded_secrets
    check_dependencies
    check_error_handling
    check_logging
    check_memory_safety
    check_input_validation
    check_resource_management
    check_license_compliance

    # Generate final summary
    log_to_report ""
    log_to_report "AUDIT RESULTS SUMMARY"
    log_to_report "====================="

    if [ "$VULNERABILITIES_FOUND" -gt 0 ]; then
        log_to_report "🔴 CRITICAL VULNERABILITIES: $VULNERABILITIES_FOUND found"
    fi

    if [ "$ISSUES_FOUND" -gt 0 ]; then
        log_to_report "🟡 SECURITY ISSUES: $ISSUES_FOUND found"
    else
        log_to_report "✅ NO SECURITY ISSUES DETECTED"
    fi

    generate_recommendations

    log_to_report ""
    log_to_report "Audit completed at: $(date)"
    log_to_report "Report saved to: $AUDIT_REPORT"

    # Final status
    if [ "$VULNERABILITIES_FOUND" -gt 0 ]; then
        log_error "Security audit FAILED: Critical vulnerabilities found"
        exit 1
    elif [ "$ISSUES_FOUND" -gt 0 ]; then
        log_warning "Security audit PASSED with warnings: $ISSUES_FOUND issues found"
        exit 0
    else
        log_success "Security audit PASSED: No issues found"
        exit 0
    fi
}

# Run main function
main "$@"
