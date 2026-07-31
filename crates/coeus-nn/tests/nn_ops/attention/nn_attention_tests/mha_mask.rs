use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Storage};
use coeus_nn::NullMask;
use coeus_tensor::Tensor;

#[test]
fn key_padding_mask_zeroes_padded_key_gradients() {
    const HEADS: usize = 2;
    const MODEL_WIDTH: usize = 8;
    const QUERY_LENGTH: usize = 3;
    const KEY_LENGTH: usize = 4;
    const HEAD_WIDTH: usize = MODEL_WIDTH / HEADS;
    const HEAD_ELEMENTS: usize = KEY_LENGTH * HEAD_WIDTH;
    const PADDED_START: usize = 2 * HEAD_WIDTH;
    const EPSILON: f32 = 1e-5;

    let backend = MoiraiBackend;
    let q = Tensor::<f32, MoiraiBackend>::ones_on([HEADS, QUERY_LENGTH, HEAD_WIDTH], &backend);
    let k = Tensor::<f32, MoiraiBackend>::ones_on([HEADS, KEY_LENGTH, HEAD_WIDTH], &backend);
    let v = Tensor::<f32, MoiraiBackend>::ones_on([HEADS, KEY_LENGTH, HEAD_WIDTH], &backend);
    let q_var = Var::new(q, true);
    let k_var = Var::new(k, true);
    let v_var = Var::new(v, true);
    let mask = Var::new(
        Tensor::<f32, MoiraiBackend>::from_slice_on(
            [1, KEY_LENGTH],
            &[1.0, 1.0, 0.0, 0.0],
            &backend,
        ),
        false,
    );

    let (output, _) = coeus_autograd::sdp_attention::<f32, MoiraiBackend, NullMask>(
        &q_var,
        &k_var,
        &v_var,
        Some(&mask),
        1.0,
    )
    .expect("valid masked attention fixture");
    coeus_autograd::sum(&output)
        .backward()
        .expect("invariant: valid attention graph completes backward");

    let gradient = k_var
        .grad
        .as_ref()
        .expect("tracked key receives a gradient")
        .read();
    let values = gradient
        .storage()
        .try_as_slice()
        .expect("Moirai storage is CPU-addressable");
    for head in values.chunks_exact(HEAD_ELEMENTS) {
        assert!(
            head[PADDED_START..]
                .iter()
                .all(|value| value.abs() < EPSILON),
            "padded key positions must have zero gradient"
        );
    }
}
