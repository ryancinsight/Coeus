
print("Attempting to import torch...")
try:
    import torch
    print(f"Torch imported: {torch.__version__}")
except ImportError as e:
    print(f"Failed to import torch: {e}")
except Exception as e:
    print(f"Error importing torch: {e}")

print("Attempting to import coeus...")
try:
    import coeus
    print(f"Coeus imported: {dir(coeus)}")
except ImportError as e:
    print(f"Failed to import coeus: {e}")
except Exception as e:
    print(f"Error importing coeus: {e}")
