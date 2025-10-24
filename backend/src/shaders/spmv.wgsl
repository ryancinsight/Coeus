struct Uniforms {
    num_rows: u32,
}

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> row_ptrs: array<u32>;
@group(0) @binding(3) var<storage, read> vec: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= uniforms.num_rows) {
        return;
    }
    
    let start = row_ptrs[row];
    let end = row_ptrs[row + 1u];
    
    var sum = 0.0;
    for (var i = start; i < end; i = i + 1u) {
        let col = col_indices[i];
        let val = values[i];
        sum = sum + val * vec[col];
    }
    
    output[row] = sum;
}
