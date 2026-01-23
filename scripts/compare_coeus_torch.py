
import io
import contextlib
import inspect
import sys
import os

def safe_import_coeus():
    try:
        import coeus
        return coeus
    except ImportError as e:
        print(f"Error importing coeus: {e}")
        return None

def get_attributes(obj, include_private=False):
    attrs = {}
    try:
        for name in dir(obj):
            if not include_private and name.startswith("_"):
                continue
            try:
                val = getattr(obj, name)
                attrs[name] = val
            except Exception:
                continue
    except Exception:
        pass
    return attrs

def compare_structures(torch_obj, coeus_obj, path="", results=None, visited=None):
    if results is None:
        results = {"missing": [], "comparable": [], "mismatch_type": []}
    if visited is None:
        visited = set()

    # Avoid infinite recursion
    if id(torch_obj) in visited:
        return results
    visited.add(id(torch_obj))

    torch_attrs = get_attributes(torch_obj)
    coeus_attrs = get_attributes(coeus_obj) if coeus_obj is not None else {}

    for name, t_val in torch_attrs.items():
        curr_path = f"{path}.{name}" if path else name
        
        # We only care about torch modules, classes, and functions primarily
        # Skip some internals
        if any(x in curr_path for x in [".testing", ".backends", ".version", ".utils.data.datapipes"]):
            continue

        if name not in coeus_attrs:
            results["missing"].append(curr_path)
            continue

        c_val = coeus_attrs[name]
        
        t_is_module = inspect.ismodule(t_val)
        c_is_module = inspect.ismodule(c_val)
        t_is_class = inspect.isclass(t_val)
        c_is_class = inspect.isclass(c_val)
        t_is_func = inspect.isfunction(t_val) or inspect.isbuiltin(t_val) or inspect.ismethod(t_val)
        c_is_func = inspect.isfunction(c_val) or inspect.isbuiltin(c_val) or inspect.ismethod(c_val)

        if t_is_module and c_is_module:
            # Recurse into submodules
            # Only if they are part of the package (avoid traversing into external deps)
            if hasattr(t_val, "__name__") and "torch" in t_val.__name__:
                 compare_structures(t_val, c_val, curr_path, results, visited)
            results["comparable"].append(f"{curr_path} (module)")
        elif t_is_class and c_is_class:
            results["comparable"].append(f"{curr_path} (class)")
        elif t_is_func and c_is_func:
            results["comparable"].append(f"{curr_path} (function)")
        else:
            # Types don't match exactly but exists
            results["mismatch_type"].append(f"{curr_path} (torch: {type(t_val).__name__}, coeus: {type(c_val).__name__})")

    return results

def compare_tensor_methods(torch_tensor_class, coeus_tensor_class):
    """Compare methods available on Tensor classes."""
    torch_methods = set()
    coeus_methods = set()
    
    for name in dir(torch_tensor_class):
        if not name.startswith("_") or name in ["__add__", "__sub__", "__mul__", "__truediv__", "__neg__"]:
            torch_methods.add(name)
    
    for name in dir(coeus_tensor_class):
        if not name.startswith("_") or name in ["__add__", "__sub__", "__mul__", "__truediv__", "__neg__"]:
            coeus_methods.add(name)
    
    common = torch_methods & coeus_methods
    missing = torch_methods - coeus_methods
    extra = coeus_methods - torch_methods
    
    return common, missing, extra

def main():
    print("Starting comparison...")
    try:
        import torch
        print(f"Torch version: {torch.__version__}")
    except ImportError:
        print("Error: torch is not installed.")
        return

    coeus = safe_import_coeus()
    if coeus is None:
        print("Coeus compilation failed or not installed properly.")
        return
    print("Coeus imported successfully.")

    results = compare_structures(torch, coeus)

    # Compare Tensor methods
    print("\n--- Tensor Method Comparison ---")
    try:
        common, missing, extra = compare_tensor_methods(torch.Tensor, coeus.Tensor)
        print(f"Common Tensor methods: {len(common)}")
        print(f"Missing Tensor methods: {len(missing)}")
        print(f"Extra Coeus methods: {len(extra)}")
    except Exception as e:
        print(f"Could not compare Tensor methods: {e}")
        common, missing, extra = set(), set(), set()

    with open("comparison_missing.txt", "w") as f:
        f.write("\\n".join(sorted(results["missing"])))
    
    with open("comparison_comparable.txt", "w") as f:
        f.write("\\n".join(sorted(results["comparable"])))

    with open("comparison_mismatch.txt", "w") as f:
        f.write("\\n".join(sorted(results["mismatch_type"])))

    # Write tensor method comparison
    with open("comparison_tensor_methods.txt", "w") as f:
        f.write("=== Common Tensor Methods ===\\n")
        f.write("\\n".join(sorted(common)))
        f.write("\\n\\n=== Missing Tensor Methods ===\\n")
        f.write("\\n".join(sorted(missing)))
        f.write("\\n\\n=== Extra Coeus Methods ===\\n")
        f.write("\\n".join(sorted(extra)))

    print(f"\\nComparison complete.")
    print(f"Module-level missing items: {len(results['missing'])}")
    print(f"Module-level comparable items: {len(results['comparable'])}")
    print(f"Type mismatches: {len(results['mismatch_type'])}")
    
    # Print summary of top-level functions in coeus
    print("\\n--- Coeus Top-Level Functions ---")
    coeus_funcs = [name for name, val in get_attributes(coeus).items() 
                   if inspect.isfunction(val) and not name.startswith("_")]
    print(f"Total: {len(coeus_funcs)}")
    print(f"Examples: {', '.join(sorted(coeus_funcs)[:20])}...")

if __name__ == "__main__":
    main()

