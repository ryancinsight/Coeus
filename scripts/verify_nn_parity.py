import coeus.nn as nn
import coeus
import sys
import traceback

def verify():
    print("Starting verification...")
    try:
        input = coeus.randn(2, 3, 4)
        print(f"Input shape: {input.shape}")

        print("Testing Flatten...")
        flatten = nn.Flatten()
        output = flatten(input)
        print(f"Flatten output shape: {output.shape}")
        
        # Check values?
        # assert output.numel() == input.numel()

        print("Testing Identity...")
        identity = nn.Identity()
        print(f"Identity created: {identity}")
        output2 = identity(input)
        print(f"Identity output shape: {output2.shape}")
        
        print("Testing Softmax...")
        softmax = nn.Softmax(dim=1)
        output3 = softmax(input)
        print(f"Softmax output shape: {output3.shape}")

        print("Testing LogSoftmax...")
        log_softmax = nn.LogSoftmax(dim=1)
        output4 = log_softmax(input)
        print(f"LogSoftmax output shape: {output4.shape}")
        
        print("Verification SUCCESS")
    except Exception as e:
        print(f"Verification FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
