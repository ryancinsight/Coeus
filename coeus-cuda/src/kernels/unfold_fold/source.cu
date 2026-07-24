#include <cuda_fp16.h>
#include <cuda_bf16.h>

struct GpuLayoutInfo {
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
};

__device__ unsigned int index3(
    const GpuLayoutInfo layout,
    unsigned int i0,
    unsigned int i1,
    unsigned int i2
) {
    return layout.offset + i0 * layout.strides[0]
        + i1 * layout.strides[1] + i2 * layout.strides[2];
}

__device__ unsigned int index4(
    const GpuLayoutInfo layout,
    unsigned int i0,
    unsigned int i1,
    unsigned int i2,
    unsigned int i3
) {
    return layout.offset + i0 * layout.strides[0]
        + i1 * layout.strides[1] + i2 * layout.strides[2]
        + i3 * layout.strides[3];
}

extern "C" __global__ void unfold1d_kernel(
    const {TYPE}* input,
    {TYPE}* output,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo output_layout,
    unsigned int kernel_size,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int total
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int out_l = output_layout.shape[2];
    const unsigned int l = idx % out_l;
    const unsigned int channel_kernel = (idx / out_l) % output_layout.shape[1];
    const unsigned int n = idx / (out_l * output_layout.shape[1]);
    const unsigned int channel = channel_kernel / kernel_size;
    const unsigned int kernel = channel_kernel % kernel_size;
    const long long source = (long long)l * stride + (long long)kernel * dilation - padding;
    const unsigned int output_index = index3(output_layout, n, channel_kernel, l);
    output[output_index] = source >= 0 && source < (long long)input_layout.shape[2]
        ? input[index3(input_layout, n, channel, (unsigned int)source)]
        : ({TYPE})0;
}

extern "C" __global__ void fold1d_kernel(
    const {TYPE}* input,
    {TYPE}* output,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo output_layout,
    unsigned int kernel_size,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int total
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int output_l = output_layout.shape[2];
    const unsigned int l = idx % output_l;
    const unsigned int channel = (idx / output_l) % output_layout.shape[1];
    const unsigned int n = idx / (output_l * output_layout.shape[1]);
    {TYPE} sum = ({TYPE})0;
    for (unsigned int kernel = 0; kernel < kernel_size; ++kernel) {
        const long long numerator = (long long)l + padding - (long long)kernel * dilation;
        if (numerator >= 0 && numerator % stride == 0) {
            const unsigned int source_l = (unsigned int)(numerator / stride);
            if (source_l < input_layout.shape[2]) {
                sum += input[index3(
                    input_layout, n, channel * kernel_size + kernel, source_l
                )];
            }
        }
    }
    output[index3(output_layout, n, channel, l)] = sum;
}

extern "C" __global__ void unfold2d_kernel(
    const {TYPE}* input,
    {TYPE}* output,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo output_layout,
    unsigned int kernel_h,
    unsigned int kernel_w,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int padding_h,
    unsigned int padding_w,
    unsigned int dilation_h,
    unsigned int dilation_w,
    unsigned int output_w,
    unsigned int total
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int locations = output_layout.shape[2];
    const unsigned int location = idx % locations;
    const unsigned int channel_kernel = (idx / locations) % output_layout.shape[1];
    const unsigned int n = idx / (locations * output_layout.shape[1]);
    const unsigned int kernel_area = kernel_h * kernel_w;
    const unsigned int channel = channel_kernel / kernel_area;
    const unsigned int kernel_offset = channel_kernel % kernel_area;
    const unsigned int kh = kernel_offset / kernel_w;
    const unsigned int kw = kernel_offset % kernel_w;
    const unsigned int oh = location / output_w;
    const unsigned int ow = location % output_w;
    const long long source_h = (long long)oh * stride_h + (long long)kh * dilation_h - padding_h;
    const long long source_w = (long long)ow * stride_w + (long long)kw * dilation_w - padding_w;
    const unsigned int output_index = index3(output_layout, n, channel_kernel, location);
    output[output_index] = source_h >= 0 && source_h < (long long)input_layout.shape[2]
        && source_w >= 0 && source_w < (long long)input_layout.shape[3]
        ? input[index4(input_layout, n, channel, (unsigned int)source_h, (unsigned int)source_w)]
        : ({TYPE})0;
}

extern "C" __global__ void fold2d_kernel(
    const {TYPE}* input,
    {TYPE}* output,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo output_layout,
    unsigned int kernel_h,
    unsigned int kernel_w,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int padding_h,
    unsigned int padding_w,
    unsigned int dilation_h,
    unsigned int dilation_w,
    unsigned int input_w,
    unsigned int total
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int output_w = output_layout.shape[3];
    const unsigned int output_h = output_layout.shape[2];
    const unsigned int w = idx % output_w;
    const unsigned int h = (idx / output_w) % output_h;
    const unsigned int channel = (idx / (output_w * output_h)) % output_layout.shape[1];
    const unsigned int n = idx / (output_w * output_h * output_layout.shape[1]);
    const unsigned int input_h = input_layout.shape[2] / input_w;
    {TYPE} sum = ({TYPE})0;
    for (unsigned int kh = 0; kh < kernel_h; ++kh) {
        const long long numerator_h = (long long)h + padding_h - (long long)kh * dilation_h;
        if (numerator_h < 0 || numerator_h % stride_h != 0) continue;
        const unsigned int source_h = (unsigned int)(numerator_h / stride_h);
        if (source_h >= input_h) continue;
        for (unsigned int kw = 0; kw < kernel_w; ++kw) {
            const long long numerator_w = (long long)w + padding_w - (long long)kw * dilation_w;
            if (numerator_w < 0 || numerator_w % stride_w != 0) continue;
            const unsigned int source_w = (unsigned int)(numerator_w / stride_w);
            if (source_w >= input_w) continue;
            const unsigned int location = source_h * input_w + source_w;
            if (location < input_layout.shape[2]) {
                const unsigned int channel_kernel =
                    (channel * kernel_h + kh) * kernel_w + kw;
                sum += input[index3(input_layout, n, channel_kernel, location)];
            }
        }
    }
    output[index4(output_layout, n, channel, h, w)] = sum;
}
