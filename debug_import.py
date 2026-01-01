
import sys
import os
import traceback

print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

try:
    print("Attempting to import _coeus...")
    from coeus import _coeus
    print("Success importing _coeus")
except ImportError:
    print("Failed to import _coeus direct")
    traceback.print_exc()
except Exception:
    traceback.print_exc()

try:
    print("\nAttempting to import coeus...")
    import coeus
    print("Success importing coeus")
except Exception:
    traceback.print_exc()
