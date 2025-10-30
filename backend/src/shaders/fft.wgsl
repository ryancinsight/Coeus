//! FFT (Fast Fourier Transform) implementation for GPU computation
//!
//! This shader implements 1D FFT using the Cooley-Tukey algorithm with
//! complex arithmetic operations using vec2<f32> representation.
//! Supports both forward and inverse transforms.

// Complex number utilities (vec2<f32> where x=real, y=imaginary)
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,  // real: a.re*b.re - a.im*b.im
        a.x * b.y + a.y * b.x   // imag: a.re*b.im + a.im*b.re
    );
}

fn complex_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

fn complex_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b;
}

fn complex_conj(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z.x, -z.y);
}

fn complex_scale(z: vec2<f32>, scale: f32) -> vec2<f32> {
    return z * scale;
}

// Compute twiddle factor: exp(-2πi * k / N) for forward FFT
// For inverse FFT: exp(+2πi * k / N)
fn twiddle(k: u32, N: u32, inverse: bool) -> vec2<f32> {
    let angle = -2.0 * 3.141592653589793 * f32(k) / f32(N);
    let sign = select(-1.0, 1.0, inverse);
    let theta = sign * angle;

    return vec2<f32>(cos(theta), sin(theta));
}

// Bit reversal permutation for index
fn bit_reverse(index: u32, N: u32) -> u32 {
    let logN = u32(log2(f32(N)));
    var reversed = 0u;
    var temp = index;

    for (var i = 0u; i < logN; i = i + 1u) {
        reversed = (reversed << 1u) | (temp & 1u);
        temp = temp >> 1u;
    }

    return reversed;
}

// 2-point DFT radix kernel
fn radix2_butterfly(
    data: ptr<storage, array<vec2<f32>>, read_write>,
    index: u32,
    stride: u32,
    N: u32,
    inverse: bool,
    inout result: array<vec2<f32>, 2>
) {
    let k = index / stride;
    let twiddle_factor = twiddle(k % (N / stride), N, inverse);

    let a = (*data)[index];
    let b = (*data)[index + stride];

    let b_twiddled = complex_mul(b, twiddle_factor);

    result[0] = complex_add(a, b_twiddled);
    result[1] = complex_sub(a, b_twiddled);
}

// 4-point DFT radix kernel
fn radix4_butterfly(
    data: ptr<storage, array<vec2<f32>>, read_write>,
    index: u32,
    stride: u32,
    N: u32,
    inverse: bool,
    inout result: array<vec2<f32>, 4>
) {
    let k = index / stride;

    let twiddle0 = twiddle(k % (N / stride), N, inverse);
    let twiddle1 = twiddle((k * 2u) % (N / stride), N, inverse);
    let twiddle2 = twiddle((k * 3u) % (N / stride), N, inverse);

    let a = (*data)[index];
    let b = (*data)[index + stride];
    let c = (*data)[index + 2u * stride];
    let d = (*data)[index + 3u * stride];

    let b_twiddled = complex_mul(b, twiddle0);
    let c_twiddled = complex_mul(c, twiddle1);
    let d_twiddled = complex_mul(d, twiddle2);

    // Radix-4 butterfly computation
    let temp0 = complex_add(a, c_twiddled);
    let temp1 = complex_add(b_twiddled, d_twiddled);
    let temp2 = complex_sub(a, c_twiddled);
    let temp3 = complex_sub(b_twiddled, d_twiddled);

    result[0] = complex_add(temp0, temp1);
    result[1] = complex_add(temp2, complex_mul(temp3, vec2<f32>(0.0, -1.0)));
    result[2] = complex_sub(temp0, temp1);
    result[3] = complex_sub(temp2, complex_mul(temp3, vec2<f32>(0.0, -1.0)));
}

// Forward 1D FFT using Cooley-Tukey algorithm
@group(0) @binding(0)
var<storage, read_write> fft_data: array<vec2<f32>>;

@group(0) @binding(1)
var<uniform> fft_params: vec3<u32>; // [N, radix, pass]

struct InverseFlag {
    inverse: u32;
};

@group(0) @binding(2)
var<uniform> inverse_flag: InverseFlag;

@compute @workgroup_size(256)
fn fft_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let N = fft_params.x;
    let radix = fft_params.y;
    let pass = fft_params.z;
    let inverse = inverse_flag.inverse != 0u;

    let thread_id = global_id.x;

    // Bit reversal permutation (first pass only)
    if (pass == 0u && thread_id < N) {
        let original_index = thread_id;
        let reversed_index = bit_reverse(original_index, N);

        if (original_index != reversed_index) {
            let temp = fft_data[original_index];
            fft_data[original_index] = fft_data[reversed_index];
            fft_data[reversed_index] = temp;
        }
    }

    // Ensure bit reversal is complete before proceeding
    storageBarrier();

    // Each pass processes different radix levels
    let stride = 1u << (pass + 1u); // 2^(pass+1)
    let block_size = 1u << (pass + 2u); // 2^(pass+2)

    if (thread_id * u32(radix) >= N) {
        return;
    }

    let block_start = (thread_id / (block_size / u32(radix))) * block_size;
    let butterfly_index = thread_id % (block_size / u32(radix));

    if (radix == 2u) {
        // 2-point DFT
        var result: array<vec2<f32>, 2>;
        radix2_butterfly(&fft_data, block_start + butterfly_index * stride / 2u, stride / 2u, N, false, result);

        fft_data[block_start + butterfly_index * stride + 0u] = result[0];
        fft_data[block_start + butterfly_index * stride + stride / 2u] = result[1];
    } else if (radix == 4u) {
        // 4-point DFT
        var result: array<vec2<f32>, 4>;
        radix4_butterfly(&fft_data, block_start + butterfly_index * stride / 4u, stride / 4u, N, false, result);

        for (var i = 0u; i < 4u; i = i + 1u) {
            fft_data[block_start + butterfly_index * stride + i * (stride / 4u)] = result[i];
        }
    }
}

// Inverse 1D FFT using Cooley-Tukey algorithm with proper scaling
@compute @workgroup_size(256)
fn fft_inverse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let N = fft_params.x;
    let radix = fft_params.y;
    let pass = fft_params.z;
    let inverse = inverse_flag.inverse != 0u;

    let thread_id = global_id.x;

    // Same algorithm as forward but with inverse twiddle factors
    if (pass == 0u && thread_id < N) {
        let original_index = thread_id;
        let reversed_index = bit_reverse(original_index, N);

        if (original_index != reversed_index) {
            let temp = fft_data[original_index];
            fft_data[original_index] = fft_data[reversed_index];
            fft_data[reversed_index] = temp;
        }
    }

    storageBarrier();

    let stride = 1u << (pass + 1u);
    let block_size = 1u << (pass + 2u);

    if (thread_id * u32(radix) >= N) {
        return;
    }

    let block_start = (thread_id / (block_size / u32(radix))) * block_size;
    let butterfly_index = thread_id % (block_size / u32(radix));

    if (radix == 2u) {
        var result: array<vec2<f32>, 2>;
        radix2_butterfly(&fft_data, block_start + butterfly_index * stride / 2u, stride / 2u, N, true, result);

        // Apply 1/N scaling for inverse FFT
        result[0] = complex_scale(result[0], 1.0 / f32(N));
        result[1] = complex_scale(result[1], 1.0 / f32(N));

        fft_data[block_start + butterfly_index * stride + 0u] = result[0];
        fft_data[block_start + butterfly_index * stride + stride / 2u] = result[1];
    } else if (radix == 4u) {
        var result: array<vec2<f32>, 4>;
        radix4_butterfly(&fft_data, block_start + butterfly_index * stride / 4u, stride / 4u, N, true, result);

        for (var i = 0u; i < 4u; i = i + 1u) {
            result[i] = complex_scale(result[i], 1.0 / f32(N));
            fft_data[block_start + butterfly_index * stride + i * (stride / 4u)] = result[i];
        }
    }
}
