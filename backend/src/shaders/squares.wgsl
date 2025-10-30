// Element-wise squaring operations (x²)
// Performs x * x for each element in the input array

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: SquareParams;

struct SquareParams {
    scale: f32,
    offset: f32,
    element_count: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.element_count) {
        return;
    }

    let x = input[index];
    let squared = x * x;

    // Apply optional scaling and offset
    let result = squared * params.scale + params.offset;

    output[index] = result;
}
