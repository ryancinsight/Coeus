use coeus_autograd::Var;
use coeus_core::Float;

/// CTC (Connectionist Temporal Classification) loss.
///
/// Computes the negative log-likelihood of a sequence labeling task where the
/// alignment between input frames and output labels is unknown (e.g. speech
/// recognition). Each sample is divided by its target length (one for empty
/// targets) before the batch mean. Empty targets retain the all-blank path.
/// Impossible alignments return infinite loss and a typed backward error.
/// The derivative is with respect to independent log probabilities; composing
/// with log-softmax produces the derivative with respect to logits.
///
/// # Arguments
/// - `log_probs` — `[T, N, C]` log-probabilities (output of `log_softmax`).
/// - `targets` — flat target indices `[sum(target_lengths)]` (no padding).
/// - `input_lengths` — `[N]` valid frame count per sample (<= T).
/// - `target_lengths` — `[N]` target sequence length per sample.
/// - `blank` — blank class index (default 0 in PyTorch).
///
/// Returns a scalar `Var` (shape `[1]`) containing the mean CTC loss.
///
/// # Errors
///
/// Returns provider errors for invalid shapes, lengths, labels, active numeric
/// values, allocation failure, or nonrepresentable arithmetic. See
/// [`coeus_autograd::ctc_loss`] for a runnable example.
pub fn ctc_loss<T: Float, B: coeus_ops::BackendOps<T> + coeus_ops::CtcOps<T> + Default>(
    log_probs: &Var<T, B>,
    targets: &[usize],
    input_lengths: &[usize],
    target_lengths: &[usize],
    blank: usize,
) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
}
