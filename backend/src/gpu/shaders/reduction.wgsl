// Reduction shader
// Performs parallel reduction (Sum, Max, Min)
// Uses workgroup shared memory for efficient reduction within workgroups.

struct Uniforms {
    op: u32, // 0: Sum, 1: Max, 2: Min
    dim: u32, // size of input array
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let index = global_id.x;
    let group_index = group_id.x;

    // Load data into shared memory
    // Handle out of bounds by loading identity
    if (index < uniforms.dim) {
        shared_data[tid] = input[index];
    } else {
        if (uniforms.op == 0u) {
            shared_data[tid] = 0.0; // Sum identity
        } else if (uniforms.op == 1u) {
            shared_data[tid] = -3.40282347e+38; // Min float (approx) for Max identity
        } else {
            shared_data[tid] = 3.40282347e+38; // Max float for Min identity
        }
    }

    workgroupBarrier();

    // Reduce within workgroup
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            if (uniforms.op == 0u) {
                shared_data[tid] = shared_data[tid] + shared_data[tid + s];
            } else if (uniforms.op == 1u) {
                shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
            } else {
                shared_data[tid] = min(shared_data[tid], shared_data[tid + s]);
            }
        }
        workgroupBarrier();
    }

    // Write result for this workgroup to output
    if (tid == 0u) {
        output[group_index] = shared_data[0];
    }
}
