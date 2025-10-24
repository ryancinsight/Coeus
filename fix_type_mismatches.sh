#!/bin/bash
# Fix type mismatches where generic T is used instead of integers/usize for shape operations

echo "Fixing type mismatches in nn/src/functional.rs..."

# Fix shape length comparisons - use 4usize, 2usize, etc.
sed -i 's/input_shape\.len() != 4/input_shape.len() != 4usize/g' nn/src/functional.rs
sed -i 's/weight_shape\.len() != 4/weight_shape.len() != 4usize/g' nn/src/functional.rs
sed -i 's/shape\.len() != 2/shape.len() != 2usize/g' nn/src/functional.rs
sed -i 's/input_shape\.len() != 2/input_shape.len() != 2usize/g' nn/src/functional.rs
sed -i 's/logits_shape\.len() != 2/logits_shape.len() != 2usize/g' nn/src/functional.rs
sed -i 's/targets_shape\.len() != 1/targets_shape.len() != 1usize/g' nn/src/functional.rs

# Fix other integer comparisons
sed -i 's/input_shape\.len() < 2/input_shape.len() < 2usize/g' nn/src/functional.rs
sed -i 's/shape\.len() < 2/shape.len() < 2usize/g' nn/src/functional.rs
sed -i 's/last_dim = input_shape\.len() - 1/last_dim = input_shape.len() - 1usize/g' nn/src/functional.rs

echo "Fixing type mismatches in nn/src/transformer.rs..."
sed -i 's/input_shape\.len() == 3/input_shape.len() == 3usize/g' nn/src/transformer.rs
sed -i 's/tgt_shape\.len() == 3/tgt_shape.len() == 3usize/g' nn/src/transformer.rs
sed -i 's/memory_shape\.len() == 3/memory_shape.len() == 3usize/g' nn/src/transformer.rs

echo "Fixing type mismatches in nn/src/functional_conv.rs..."
sed -i 's/input_shape\.len() != 4/input_shape.len() != 4usize/g' nn/src/functional_conv.rs
sed -i 's/weight_shape\.len() != 4/weight_shape.len() != 4usize/g' nn/src/functional_conv.rs

echo "Fixing type mismatches in nn/src/loss/*.rs..."
sed -i 's/log_probs_shape\.len() != 2/log_probs_shape.len() != 2usize/g' nn/src/loss/nll.rs
sed -i 's/targets_shape\.len() != 1/targets_shape.len() != 1usize/g' nn/src/loss/nll.rs

echo "Fixing type mismatches in nn/src/upsample.rs..."
sed -i 's/(input_h - 1)/(input_h - 1usize)/g' nn/src/upsample.rs
sed -i 's/(output_h - 1)/(output_h - 1usize)/g' nn/src/upsample.rs
sed -i 's/(input_w - 1)/(input_w - 1usize)/g' nn/src/upsample.rs
sed -i 's/(output_w - 1)/(output_w - 1usize)/g' nn/src/upsample.rs
sed -i 's/input_h - 1/input_h - 1usize/g' nn/src/upsample.rs
sed -i 's/input_w - 1/input_w - 1usize/g' nn/src/upsample.rs

echo "Fixing type mismatches in nn/src/init.rs..."
sed -i 's/shape\.len() < 2/shape.len() < 2usize/g' nn/src/init.rs
sed -i 's/shape\.len() - 2/shape.len() - 2usize/g' nn/src/init.rs
sed -i 's/shape\.len() - 1/shape.len() - 1usize/g' nn/src/init.rs

echo "Type mismatch fixes complete"
