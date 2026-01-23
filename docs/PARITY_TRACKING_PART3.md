# File Structure for Parity Tracking (Continued)

### Script 2: Generate Parity Report

```bash
#!/bin/bash
# scripts/generate_parity_report.sh

OUTPUT="docs/BACKEND_PARITY_REPORT.md"

cat > $OUTPUT << 'EOF'
# Backend Implementation Parity Report

Generated: $(date)

## Summary

| Operation Category | CPU | GPU | TPU | NPU |
|-------------------|-----|-----|-----|-----|
EOF

# For each category
for category in arithmetic linear_algebra activation reduction; do
    echo -n "| $category | " >> $OUTPUT
    
    for backend in cpu gpu tpu npu; do
        COUNT=$(find backend/src/$backend/$category -name '*.rs' -not -name 'mod.rs' 2>/dev/null | wc -l)
        echo -n "$COUNT | " >> $OUTPUT
    done
    
    echo "" >> $OUTPUT
done

cat >> $OUTPUT << 'EOF'

## Detailed Status

EOF

# For each operation file
for file in $(find backend/src/cpu -name '*.rs' -not -name 'mod.rs' -not -name 'backend.rs' | sort); do
    REL_PATH=$(echo $file | sed 's|backend/src/cpu/||')
    OP_NAME=$(basename $file .rs)
    CATEGORY=$(dirname $REL_PATH)
    
    echo "### $CATEGORY/$OP_NAME" >> $OUTPUT
    echo "" >> $OUTPUT
    echo "| Backend | Status | File |" >> $OUTPUT
    echo "|---------|--------|------|" >> $OUTPUT
    
    for backend in cpu gpu tpu npu; do
        FILE_PATH="backend/src/$backend/$REL_PATH"
        if [ -f "$FILE_PATH" ]; then
            echo "| $backend | ✅ Implemented | \`$FILE_PATH\` |" >> $OUTPUT
        else
            echo "| $backend | ❌ Missing | - |" >> $OUTPUT
        fi
    done
    
    echo "" >> $OUTPUT
done

echo "Parity report generated: $OUTPUT"
```

### Script 3: Check API Consistency

```bash
#!/bin/bash
# scripts/check_api_consistency.sh

echo "Checking API consistency across backends..."

for file in $(find backend/src/cpu -name '*.rs' -not -name 'mod.rs' -not -name 'backend.rs'); do
    REL_PATH=$(echo $file | sed 's|backend/src/cpu/||')
    OP_NAME=$(basename $file .rs)
    
    # Extract function signature from CPU implementation
    CPU_SIG=$(grep -A 3 "pub fn ${OP_NAME}_primitive" $file | head -4)
    
    # Check GPU implementation
    GPU_FILE="backend/src/gpu/$REL_PATH"
    if [ -f "$GPU_FILE" ]; then
        GPU_SIG=$(grep -A 3 "pub fn ${OP_NAME}_primitive" $GPU_FILE | head -4)
        
        if [ "$CPU_SIG" != "$GPU_SIG" ]; then
            echo "⚠️  API mismatch in $REL_PATH"
            echo "  CPU: $CPU_SIG"
            echo "  GPU: $GPU_SIG"
        fi
    fi
done
```

## Benefits of This Structure

### 1. Automated Parity Tracking

Scripts can automatically detect:
- Missing implementations
- Extra implementations
- API inconsistencies
- Coverage gaps

### 2. Clear Implementation Status

File existence = Implementation status:
- File exists → Operation implemented
- File missing → Operation not implemented
- No ambiguity

### 3. Easy Navigation

Developers can quickly find implementations:
```
backend/src/gpu/activation/relu.rs  ← GPU ReLU implementation
backend/src/cpu/activation/relu.rs  ← CPU ReLU implementation
```

### 4. Parallel Development

Multiple developers can work on different backends simultaneously:
- Backend A developer: Implements `backend/src/gpu/arithmetic/add.rs`
- Backend B developer: Implements `backend/src/tpu/arithmetic/add.rs`
- No conflicts, clear separation

### 5. Testing Parity

Test structure mirrors implementation structure:
```
tests/backend/
├── cpu/
│   └── arithmetic/
│       └── test_add.rs
├── gpu/
│   └── arithmetic/
│       └── test_add.rs
└── tpu/
    └── arithmetic/
        └── test_add.rs
```
