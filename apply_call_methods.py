import os

file_path = r'd:\coeus\pycoeus\src\nn.rs'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Check if the line is a forward method signature
    # Case 1: forward(&self, input: &PyTensor) -> PyResult<PyTensor>
    if 'fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {' in line:
        indent = line[:line.find('fn')]
        new_lines.append(f"{indent}fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {{\n")
        new_lines.append(f"{indent}    self.forward(input)\n")
        new_lines.append(f"{indent}}}\n\n")
    # Case 2: RNN/GRU forward
    elif 'fn forward(&self, input: &PyTensor, hidden: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {' in line:
        indent = line[:line.find('fn')]
        new_lines.append(f"{indent}fn __call__(&self, input: &PyTensor, hidden: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {{\n")
        new_lines.append(f"{indent}    self.forward(input, hidden)\n")
        new_lines.append(f"{indent}}}\n\n")
    # Case 3: LSTM forward
    elif 'fn forward(&self, input: &PyTensor, state: Option<(PyRef<PyTensor>, PyRef<PyTensor>)>) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {' in line:
        indent = line[:line.find('fn')]
        new_lines.append(f"{indent}fn __call__(&self, input: &PyTensor, state: Option<(PyRef<PyTensor>, PyRef<PyTensor>)>) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {{\n")
        new_lines.append(f"{indent}    self.forward(input, state)\n")
        new_lines.append(f"{indent}}}\n\n")
    
    new_lines.append(line)

with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(new_lines)
