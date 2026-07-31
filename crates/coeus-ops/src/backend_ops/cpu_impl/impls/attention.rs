use super::super::CpuBackend;
use crate::backend_ops::traits::{AttentionOps, AttentionScalar};
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};
use coeus_leto::{
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
    AttentionBackward, AttentionForward, AttentionGradientTargets, ReadOperand, WriteOperand,
};

fn map_attention_error(operation: &'static str, error: leto_ops::AttentionError) -> BackendError {
    match error {
        leto_ops::AttentionError::Shape {
            expected, actual, ..
        } => BackendError::ShapeMismatch {
            operation,
            lhs: actual.to_vec(),
            rhs: expected.to_vec(),
        },
        leto_ops::AttentionError::MaskShape { actual, target } => {
            BackendError::IncompatibleBroadcast {
                operation,
                from: actual.to_vec(),
                to: target.to_vec(),
            }
        }
        leto_ops::AttentionError::WorkspaceOverflow => BackendError::Overflow {
            operation,
            reason: "attention workspace size overflow",
        },
        other => BackendError::Storage {
            operation,
            reason: other.to_string(),
        },
    }
}

impl<T: Scalar + leto_ops::Scalar + coeus_leto::AttentionScalar, B: CpuBackend> AttentionOps<T>
    for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    #[inline]
    fn sdp_attention(
        &self,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        key_padding_mask: Option<&Self::DeviceBuffer<T>>,
        key_padding_mask_layout: Option<&Layout>,
        is_causal: bool,
        scale: T,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
        attn_weights: &mut Self::DeviceBuffer<T>,
        attn_weights_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: AttentionScalar,
    {
        let keep_mask = match (key_padding_mask, key_padding_mask_layout) {
            (None, None) => None,
            (Some(data), Some(layout)) if matches!(layout.ndim(), 1 | 2) => Some(ReadOperand {
                layout,
                data: data.as_slice(),
            }),
            (Some(_), Some(layout)) => {
                return Err(BackendError::UnsupportedRank {
                    operation: "attention forward",
                    rank: layout.ndim(),
                    max_rank: 2,
                });
            }
            _ => {
                return Err(BackendError::Storage {
                    operation: "attention forward",
                    reason: "key-padding mask storage and layout must be supplied together".into(),
                });
            }
        };
        scaled_dot_product_attention_into(AttentionForward {
            query: ReadOperand {
                layout: query_layout,
                data: query.as_slice(),
            },
            key: ReadOperand {
                layout: key_layout,
                data: key.as_slice(),
            },
            value: ReadOperand {
                layout: value_layout,
                data: value.as_slice(),
            },
            keep_mask,
            is_causal,
            scale,
            output: WriteOperand {
                layout: output_layout,
                data: output.as_mut_slice(),
            },
            weights: WriteOperand {
                layout: attn_weights_layout,
                data: attn_weights.as_mut_slice(),
            },
        })
        .map_err(|error| map_attention_error("attention forward", error))
    }

    #[inline]
    fn sdp_attention_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        attn_weights: &Self::DeviceBuffer<T>,
        attn_weights_layout: &Layout,
        scale: T,
        grad_q: Option<(&mut Self::DeviceBuffer<T>, &Layout)>,
        grad_k: Option<(&mut Self::DeviceBuffer<T>, &Layout)>,
        grad_v: Option<(&mut Self::DeviceBuffer<T>, &Layout)>,
    ) -> Result<(), Self::Error>
    where
        T: AttentionScalar,
    {
        scaled_dot_product_attention_backward_accumulate(AttentionBackward {
            output_gradient: ReadOperand {
                layout: grad_out_layout,
                data: grad_out.as_slice(),
            },
            query: ReadOperand {
                layout: query_layout,
                data: query.as_slice(),
            },
            key: ReadOperand {
                layout: key_layout,
                data: key.as_slice(),
            },
            value: ReadOperand {
                layout: value_layout,
                data: value.as_slice(),
            },
            weights: ReadOperand {
                layout: attn_weights_layout,
                data: attn_weights.as_slice(),
            },
            scale,
            gradients: AttentionGradientTargets {
                query: grad_q.map(|(data, layout)| WriteOperand {
                    layout,
                    data: data.as_mut_slice(),
                }),
                key: grad_k.map(|(data, layout)| WriteOperand {
                    layout,
                    data: data.as_mut_slice(),
                }),
                value: grad_v.map(|(data, layout)| WriteOperand {
                    layout,
                    data: data.as_mut_slice(),
                }),
            },
        })
        .map_err(|error| map_attention_error("attention backward", error))
    }
}
