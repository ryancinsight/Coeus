import sys
import os
import torch
import numpy as np

# Ensure we can import locally if needed, though running in venv should use installed package
# sys.path.insert(0, os.path.abspath("pycoeus/python"))

def verify():
    print("Starting verification of Coeus ConvTranspose3d Binding... and Conv2d")
    
    try:
        import coeus
        print(f"Coeus package found: {coeus.__file__}")
        
        try:
             print(f"Coeus nn package: {coeus.nn.__file__}")
        except:
             print("coeus.nn not importable directly??")

        # Check content of _coeus
        import coeus._coeus as _c
        print(f"Items in coeus._coeus: {dir(_c)}")

        if hasattr(_c, 'Conv2d'):
             print("Conv2d found in _coeus.")
        else:
             print("Conv2d NOT found in _coeus.")

        if hasattr(_c, 'ConvTranspose3d'):
             print("ConvTranspose3d found in _coeus!")
        else:
             print("ConvTranspose3d NOT found in _coeus.")
             # If missing, we can't proceed with instantiation
             return

        # Instantiate
        in_channels = 3
        out_channels = 16
        kernel_size = (3, 3, 3)
        stride = (1, 1, 1)
        padding = (0, 0, 0)
        output_padding = (0, 0, 0)
        
        print(f"Instantiating ConvTranspose3d({in_channels}, {out_channels}, {kernel_size})...")
        ct3d = _c.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        print("Success!")
        
        # Verify attributes
        if hasattr(ct3d, 'weight') and hasattr(ct3d, 'bias'):
             print("Attributes weight and bias exist.")
        
        # Verify weight shape?
        # w = ct3d.weight
        # print(f"Weight shape: {w.shape}")

    except ImportError as e:
        print(f"ImportError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
