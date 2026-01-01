import sys
try:
    import coeus
    import coeus._coeus
    print("coeus._coeus exports:", dir(coeus._coeus))
except ImportError as e:
    print(f"ImportError: {e}")
