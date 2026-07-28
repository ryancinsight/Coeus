use super::{AttentionMask, AttentionShape, checked_attention_dimensions};

fn shape() -> AttentionShape {
    AttentionShape {
        batch: 2,
        seq_q: 3,
        seq_k: 4,
        d_k: 5,
        d_v: 6,
    }
}

#[test]
fn accepts_contiguous_rank_two_mask_dimensions() {
    let dimensions = checked_attention_dimensions(
        shape(),
        AttentionMask {
            has_mask: true,
            ndim: 2,
            num_heads: 1,
        },
    )
    .expect("valid attention dimensions");

    assert_eq!(dimensions.mask_elements, 8);
    assert_eq!(dimensions.attention_elements, 24);
    assert_eq!(dimensions.total_q, 6);
    assert_eq!(dimensions.total_k, 8);
}

#[test]
fn rejects_zero_dimensions_and_inconsistent_mask_rank() {
    let mut zero_shape = shape();
    zero_shape.seq_q = 0;
    assert!(
        checked_attention_dimensions(
            zero_shape,
            AttentionMask {
                has_mask: false,
                ndim: 0,
                num_heads: 1,
            },
        )
        .is_none()
    );

    assert!(
        checked_attention_dimensions(
            shape(),
            AttentionMask {
                has_mask: false,
                ndim: 1,
                num_heads: 1,
            },
        )
        .is_none()
    );
}

#[test]
fn rejects_non_divisible_mask_heads_and_product_overflow() {
    assert!(
        checked_attention_dimensions(
            shape(),
            AttentionMask {
                has_mask: true,
                ndim: 2,
                num_heads: 3,
            },
        )
        .is_none()
    );

    let overflow_shape = AttentionShape {
        batch: usize::MAX,
        ..shape()
    };
    assert!(
        checked_attention_dimensions(
            overflow_shape,
            AttentionMask {
                has_mask: false,
                ndim: 0,
                num_heads: 1,
            },
        )
        .is_none()
    );
}
