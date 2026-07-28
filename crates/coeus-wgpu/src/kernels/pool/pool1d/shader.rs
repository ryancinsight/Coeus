use crate::backend::{WgpuBackendError, WgpuScalar};
use coeus_core::BackendError;

#[derive(Clone, Copy)]
pub(super) enum PoolKind {
    MaxForward,
    MaxBackward,
    AvgForward,
    AvgBackward,
}

/// Forward dispatch cannot select a backward shader.
#[derive(Clone, Copy)]
pub(super) enum ForwardPoolKind {
    Max,
    Avg,
}

impl From<ForwardPoolKind> for PoolKind {
    #[inline]
    fn from(kind: ForwardPoolKind) -> Self {
        match kind {
            ForwardPoolKind::Max => Self::MaxForward,
            ForwardPoolKind::Avg => Self::AvgForward,
        }
    }
}

pub(super) fn parameter(value: usize, name: &str) -> Result<u32, WgpuBackendError> {
    u32::try_from(value).map_err(|_| {
        WgpuBackendError::Validation(BackendError::Storage {
            operation: "pool1d",
            reason: format!("{name} value {value} exceeds the WGSL u32 ABI"),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::parameter;
    use crate::backend::WgpuBackendError;
    use coeus_core::BackendError;

    #[test]
    fn accepts_parameters_representable_by_the_wgsl_abi() {
        assert!(matches!(parameter(17, "stride"), Ok(17)));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_parameters_outside_the_wgsl_abi() {
        let value = usize::try_from(u64::from(u32::MAX) + 1).expect("u64 value fits usize");

        assert!(matches!(
            parameter(value, "stride"),
            Err(WgpuBackendError::Validation(BackendError::Storage {
                operation: "pool1d",
                reason,
            })) if reason == "stride value 4294967296 exceeds the WGSL u32 ABI"
        ));
    }
}

pub(super) fn shader_source<T: WgpuScalar>(kind: PoolKind) -> String {
    let bindings = match kind {
        PoolKind::MaxForward | PoolKind::AvgForward => {
            r#"
            @group(0) @binding(0) var<storage, read> input: array<{TYPE}>;
            @group(0) @binding(1) var<storage, read_write> output: array<{TYPE}>;
            @group(0) @binding(2) var<storage, read> input_layout: LayoutInfo;
            @group(0) @binding(3) var<storage, read> output_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> params: array<u32, 4>;
            "#
        }
        PoolKind::MaxBackward => {
            r#"
            @group(0) @binding(0) var<storage, read> grad_out: array<{TYPE}>;
            @group(0) @binding(1) var<storage, read> input: array<{TYPE}>;
            @group(0) @binding(2) var<storage, read_write> grad_input: array<{TYPE}>;
            @group(0) @binding(3) var<storage, read> grad_out_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> input_layout: LayoutInfo;
            @group(0) @binding(5) var<storage, read> grad_input_layout: LayoutInfo;
            @group(0) @binding(6) var<storage, read> params: array<u32, 4>;
            "#
        }
        PoolKind::AvgBackward => {
            r#"
            @group(0) @binding(0) var<storage, read> grad_out: array<{TYPE}>;
            @group(0) @binding(1) var<storage, read_write> grad_input: array<{TYPE}>;
            @group(0) @binding(2) var<storage, read> grad_out_layout: LayoutInfo;
            @group(0) @binding(3) var<storage, read> grad_input_layout: LayoutInfo;
            @group(0) @binding(4) var<storage, read> params: array<u32, 4>;
            "#
        }
    };
    let body = match kind {
        PoolKind::MaxForward => {
            r#"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let out_l = output_layout.shape[2];
                let channels = output_layout.shape[1];
                let total = output_layout.shape[0] * channels * out_l;
                if (idx >= total) { return; }
                let position_out = idx % out_l;
                let channel = (idx / out_l) % channels;
                let batch = idx / (out_l * channels);
                var maximum: {TYPE} = {ZERO};
                var found = false;
                for (var window: u32 = 0u; window < params[0]; window = window + 1u) {
                    let position_in = i32(position_out) * i32(params[1])
                        + i32(window) * i32(params[3]) - i32(params[2]);
                    if (position_in >= 0 && u32(position_in) < input_layout.shape[2]) {
                        let value = input[index3(input_layout, batch, channel, u32(position_in))];
                        if (!found || value > maximum) {
                            maximum = value;
                            found = true;
                        }
                    }
                }
                output[index3(output_layout, batch, channel, position_out)] =
                    select({ZERO}, maximum, found);
            }
            "#
        }
        PoolKind::MaxBackward => {
            r#"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let in_l = grad_input_layout.shape[2];
                let channels = grad_input_layout.shape[1];
                let total = grad_input_layout.shape[0] * channels * in_l;
                if (idx >= total) { return; }
                let position_in = idx % in_l;
                let channel = (idx / in_l) % channels;
                let batch = idx / (in_l * channels);
                var gradient: {TYPE} = {ZERO};
                for (var window: u32 = 0u; window < params[0]; window = window + 1u) {
                    let numerator = i32(position_in) + i32(params[2])
                        - i32(window) * i32(params[3]);
                    if (numerator < 0 || numerator % i32(params[1]) != 0) { continue; }
                    let position_out = u32(numerator / i32(params[1]));
                    if (position_out >= grad_out_layout.shape[2]) { continue; }
                    var maximum: {TYPE} = {ZERO};
                    var maximum_position = 0u;
                    var found = false;
                    for (var candidate: u32 = 0u; candidate < params[0]; candidate = candidate + 1u) {
                        let source = i32(position_out) * i32(params[1])
                            + i32(candidate) * i32(params[3]) - i32(params[2]);
                        if (source >= 0 && u32(source) < input_layout.shape[2]) {
                            let value = input[index3(input_layout, batch, channel, u32(source))];
                            if (!found || value > maximum) {
                                maximum = value;
                                maximum_position = u32(source);
                                found = true;
                            }
                        }
                    }
                    if (found && maximum_position == position_in) {
                        gradient = gradient + grad_out[index3(
                            grad_out_layout, batch, channel, position_out
                        )];
                    }
                }
                let index = index3(grad_input_layout, batch, channel, position_in);
                grad_input[index] = grad_input[index] + gradient;
            }
            "#
        }
        PoolKind::AvgForward => {
            r#"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let out_l = output_layout.shape[2];
                let channels = output_layout.shape[1];
                let total = output_layout.shape[0] * channels * out_l;
                if (idx >= total) { return; }
                let position_out = idx % out_l;
                let channel = (idx / out_l) % channels;
                let batch = idx / (out_l * channels);
                var sum: {TYPE} = {ZERO};
                var count = 0u;
                for (var window: u32 = 0u; window < params[0]; window = window + 1u) {
                    let position_in = i32(position_out) * i32(params[1])
                        + i32(window) * i32(params[3]) - i32(params[2]);
                    if (position_in >= 0 && u32(position_in) < input_layout.shape[2]) {
                        sum = sum + input[index3(input_layout, batch, channel, u32(position_in))];
                        count = count + 1u;
                    }
                }
                output[index3(output_layout, batch, channel, position_out)] =
                    select({ZERO}, sum / {TYPE}(count), count != 0u);
            }
            "#
        }
        PoolKind::AvgBackward => {
            r#"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let in_l = grad_input_layout.shape[2];
                let channels = grad_input_layout.shape[1];
                let total = grad_input_layout.shape[0] * channels * in_l;
                if (idx >= total) { return; }
                let position_in = idx % in_l;
                let channel = (idx / in_l) % channels;
                let batch = idx / (in_l * channels);
                var gradient: {TYPE} = {ZERO};
                for (var window: u32 = 0u; window < params[0]; window = window + 1u) {
                    let numerator = i32(position_in) + i32(params[2])
                        - i32(window) * i32(params[3]);
                    if (numerator < 0 || numerator % i32(params[1]) != 0) { continue; }
                    let position_out = u32(numerator / i32(params[1]));
                    if (position_out >= grad_out_layout.shape[2]) { continue; }
                    var count = 0u;
                    for (var candidate: u32 = 0u; candidate < params[0]; candidate = candidate + 1u) {
                        let source = i32(position_out) * i32(params[1])
                            + i32(candidate) * i32(params[3]) - i32(params[2]);
                        if (source >= 0 && u32(source) < grad_input_layout.shape[2]) {
                            count = count + 1u;
                        }
                    }
                    if (count != 0u) {
                        gradient = gradient + grad_out[index3(
                            grad_out_layout, batch, channel, position_out
                        )] / {TYPE}(count);
                    }
                }
                let index = index3(grad_input_layout, batch, channel, position_in);
                grad_input[index] = grad_input[index] + gradient;
            }
            "#
        }
    };

    format!(
        r#"
        struct LayoutInfo {{
            offset: u32,
            ndim: u32,
            shape: array<u32, 8>,
            strides: array<u32, 8>,
        }}

        {BINDINGS}

        fn index3(ly: LayoutInfo, i0: u32, i1: u32, i2: u32) -> u32 {{
            var index = ly.offset;
            if (ly.ndim > 0u) {{ index = index + i0 * ly.strides[0]; }}
            if (ly.ndim > 1u) {{ index = index + i1 * ly.strides[1]; }}
            if (ly.ndim > 2u) {{ index = index + i2 * ly.strides[2]; }}
            return index;
        }}

        {BODY}
        "#,
        BINDINGS = bindings,
        BODY = body,
    )
    .replace("{TYPE}", T::WGSL_TYPE)
    .replace("{ZERO}", T::WGSL_ZERO)
}
