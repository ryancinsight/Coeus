try:
    import coeus
    print("Top-level coeus import successful")
except ImportError as e:
    print(f"Import coeus failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from coeus import nn
    print("coeus.nn import successful")
except ImportError as e:
    print(f"Import coeus.nn failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from coeus import optim
    print("coeus.optim import successful")
except ImportError as e:
    print(f"Import coeus.optim failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from coeus import functional
    print("coeus.functional import successful")
except ImportError as e:
    print(f"Import coeus.functional failed: {e}")
    import traceback
    traceback.print_exc()
