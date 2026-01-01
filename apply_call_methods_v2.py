import os
import re

file_path = r'd:\coeus\pycoeus\src\nn.rs'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Improved pattern matching for forward methods, possibly multi-line
patterns = [
    (r'fn\s+forward\s*\(\s*&self\s*,\s*input\s*:\s*&PyTensor\s*\)\s*->\s*PyResult\s*<\s*PyTensor\s*>\s*\{', 
     'fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {\n        self.forward(input)\n    }'),
    (r'fn\s+forward\s*\(\s*&self\s*,\s*input\s*:\s*&PyTensor\s*,\s*hidden\s*:\s*Option\s*<\s*&PyTensor\s*>\s*\)\s*->\s*PyResult\s*<\s*\(\s*PyTensor\s*,\s*PyTensor\s*\)\s*>\s*\{',
     'fn __call__(&self, input: &PyTensor, hidden: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {\n        self.forward(input, hidden)\n    }'),
    (r'fn\s+forward\s*\(\s*&self\s*,\s*input\s*:\s*&PyTensor\s*,\s*state\s*:\s*Option\s*<\s*\(\s*PyRef\s*<\s*PyTensor\s*>\s*,\s*PyRef\s*<\s*PyTensor\s*>\s*\)\s*>\s*\)\s*->\s*PyResult\s*<\s*\(\s*PyTensor\s*,\s*\(\s*PyTensor\s*,\s*PyTensor\s*\)\s*\)\s*>\s*\{',
     'fn __call__(&self, input: &PyTensor, state: Option<(PyRef<PyTensor>, PyRef<PyTensor>)>) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {\n        self.forward(input, state)\n    }')
]

for pattern, call_impl in patterns:
    matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
    offset = 0
    for match in matches:
        start = match.start() + offset
        # Find indent of the line where match starts
        last_newline = content.rfind('\n', 0, start)
        indent_match = re.search(r'^(\s*)', content[last_newline+1:], re.MULTILINE)
        indent = indent_match.group(1) if indent_match else '    '
        
        # Check if __call__ is already there (within 200 chars before)
        prev_block = content[max(0, start-200):start]
        if 'fn __call__' in prev_block:
            continue
            
        full_call_impl = f"{indent}{call_impl}\n\n"
        content = content[:start] + full_call_impl + content[start:]
        offset += len(full_call_impl)

with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)
