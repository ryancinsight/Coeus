fn repeated(groups: &[f32]) -> Vec<f32> {
    let mut values = Vec::with_capacity(groups.len() * 8);
    for &group in groups {
        values.extend([group; 8]);
    }
    values
}

pub(super) fn mha_out() -> Vec<f32> {
    repeated(&[
        5.160251, 0.535744, 6.037920, -8.599031, 11.040882, -9.066002,
    ])
}

pub(super) fn mha_dq() -> Vec<f32> {
    repeated(&[3.055613, 5.645340, 1.507951, 1.367525, 0.041598, 0.680567])
}

pub(super) fn mha_dk() -> Vec<f32> {
    repeated(&[
        -2.527703, 0.197418, 2.330285, -3.211075, -0.106984, 3.318061,
    ])
}

pub(super) fn mha_dv() -> Vec<f32> {
    repeated(&[6.522905, 5.600709, 10.916387, 3.288871, 7.702006, 12.049126])
}

pub(super) fn mha_dwq() -> Vec<f32> {
    let mut values = repeated(&[
        -1.034660, -1.420694, -1.806727, -2.192761, -2.578795, -2.964828,
    ]);
    values.extend([-3.350863; 6]);
    values.extend([-3.350862; 2]);
    values.extend([-3.736896; 8]);
    values
}

pub(super) fn mha_dbq() -> Vec<f32> {
    repeated(&[15.373242])
}

pub(super) fn mha_dwk() -> Vec<f32> {
    let mut values = repeated(&[
        -0.579298, -0.617349, -0.655399, -0.693449, -0.731500, -0.769550,
    ]);
    values.extend([-0.807601; 6]);
    values.extend([-0.807602; 2]);
    values.extend([-0.845651; 8]);
    values
}

pub(super) fn mha_dbk() -> Vec<f32> {
    repeated(&[0.000001])
}

pub(super) fn mha_dwv() -> Vec<f32> {
    let mut values = repeated(&[-0.183809, -0.008701, 0.166407]);
    values.extend([0.341516; 6]);
    values.extend([0.341515; 2]);
    values.extend([0.516623; 8]);
    values.extend([0.691731; 6]);
    values.extend([0.691730; 2]);
    values.extend([0.866839; 8]);
    values.extend([1.041946; 6]);
    values.extend([1.041947; 2]);
    values
}

pub(super) fn mha_dbv() -> Vec<f32> {
    repeated(&[19.200001])
}

pub(super) fn mha_dwo() -> Vec<f32> {
    repeated(&[1.221802; 8])
}

pub(super) fn mha_dbo() -> Vec<f32> {
    repeated(&[6.0])
}

pub(super) fn transpose_8x8(src: &[f32]) -> Vec<f32> {
    assert_eq!(src.len(), 64);
    let mut dst = vec![0.0f32; 64];
    for r in 0..8 {
        for c in 0..8 {
            dst[c * 8 + r] = src[r * 8 + c];
        }
    }
    dst
}
