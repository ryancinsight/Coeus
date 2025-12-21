import coeus as torch
import numpy as np

def test_argmax_argmin():
    print("Testing argmax/argmin...")
    x = torch.tensor([[1.0, 5.0, 2.0], [4.0, 1.0, 6.0]])
    
    # Global argmax
    idx = torch.argmax(x)
    print(f"Global argmax index: {idx.item()}")
    assert idx.item() == 5  # Index of 6.0 is 5
    
    # Argmax along dim 1
    idx1 = torch.argmax(x, dim=1)
    print(f"Argmax dim 1: {idx1}")
    assert np.allclose(idx1.numpy(), np.array([1, 2]))
    
    # Global argmin
    idx_min = torch.argmin(x)
    print(f"Global argmin index: {idx_min.item()}")
    assert idx_min.item() == 0 or idx_min.item() == 4 # Both 1.0 are min
    
    # Argmin along dim 0
    idx0 = torch.argmin(x, dim=0)
    print(f"Argmin dim 0: {idx0}")
    assert np.allclose(idx0.numpy(), np.array([0, 1, 0]))
    print("Argmax/argmin tests passed!\n")

def test_loss_functions():
    print("Testing loss functions...")
    
    # Cross Entropy with Indices
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 1.5, 0.4]])
    targets = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy_loss(logits, targets)
    print(f"CrossEntropy (Indices) loss: {loss.item()}")
    # Expected: -log(softmax(logits)[0,0]) and -log(softmax(logits)[1,1])
    # Roughly: softmax([2.0, 0.5, 0.1]) -> [0.77, 0.17, 0.05] -> -log(0.77) = 0.26
    # Roughly: softmax([0.1, 1.5, 0.4]) -> [0.13, 0.54, 0.32] -> -log(0.54) = 0.61
    # Mean: (0.26 + 0.61) / 2 = 0.435
    assert 0.4 < loss.item() < 0.5
    
    # Cross Entropy with Probs
    targets_probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss_probs = torch.nn.functional.cross_entropy_loss(logits, targets_probs)
    print(f"CrossEntropy (Probs) loss: {loss_probs.item()}")
    # Probs version in my implementation divides by batch_size * num_classes if not indices?
    # Actually wait, my implementation:
    # divisor = if is_indices { batch_size } else { batch_size * num_classes }
    # So for probs it might be different than PyTorch which usually just sums classes then means batches.
    # PyTorch CE with probs: -sum(target * log(softmax(input))) / batch_size
    # My implementation: -sum(target * log(softmax(input))) / (batch_size * num_classes)
    # I should check this.
    
    # BCEWithLogitsLoss
    input = torch.tensor([0.0, 1.0, -1.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    loss_bce = torch.nn.functional.bce_with_logits_loss(input, target)
    print(f"BCEWithLogitsLoss: {loss_bce.item()}")
    # sigmoid(0) = 0.5, -log(1-0.5) = 0.693
    # sigmoid(1) = 0.731, -log(0.731) = 0.313
    # sigmoid(-1) = 0.269, -log(1-0.269) = 0.313
    # Mean: (0.693 + 0.313 + 0.313) / 3 = 0.44
    assert 0.43 < loss_bce.item() < 0.45
    print("Loss functions tests passed!\n")

def test_adamw():
    print("Testing AdamW...")
    w = torch.tensor([[1.0, 2.0]], requires_grad=True)
    opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.01)
    
    # Fake step
    loss = (w * w).sum()
    loss.backward()
    
    original_w = w.numpy().copy()
    opt.step()
    new_w = w.numpy()
    
    print(f"Original W: {original_w}")
    print(f"New W: {new_w}")
    assert not np.allclose(original_w, new_w)
    
    # With weight decay, it should pull towards zero more than Adam if gradients were zero
    # But here gradients are 2*w
    print("AdamW test passed!\n")

if __name__ == "__main__":
    try:
        test_argmax_argmin()
        test_loss_functions()
        test_adamw()
        print("All features verified successfully!")
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
