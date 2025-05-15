# Linear Model PyTorch Scripts

This repository contains three Python scripts that implement a simple linear model using PyTorch, perform inference, and export the model to ONNX format. Each script is designed to be straightforward and demonstrates different configurations of a linear neural network.

## Scripts Overview

### 1. `linear_model.py`
- **Purpose**: Implements a single linear layer model with custom weights, performs inference, and exports the model to ONNX.
- **Functionality**: 
  - Defines a single-layer linear model with 4 input features and 3 output features using custom weights.
  - Performs inference with input tensor `[1, 2, 3, 4]` and prints the result.
  - Exports the model to ONNX format with dynamic batch size support, saved to `models/example.onnx`.
- **Key Features**:
  - Single linear layer with optional bias (default: disabled).
  - Dynamic batch size support in ONNX export.

### 2. `linear_two_head_model.py`
- **Purpose**: Implements a model with two parallel linear layers, performs inference, and exports to ONNX.
- **Functionality**:
  - Defines a two-head linear model with two parallel linear layers, each with 4 input features and 3 output features, using different custom weights.
  - Performs inference with input tensor `[1, 2, 3, 4]` and prints both outputs.
  - Exports the model to ONNX format, saved to `models/example_two_head.onnx`.
- **Key Features**:
  - Two parallel linear layers with separate weights.
  - ONNX export with multiple outputs (`output0`, `output1`).

### 3. `linear_model_dynamic.py`
- **Purpose**: Similar to `linear_model.py`, but emphasizes dynamic shape support in ONNX export.
- **Functionality**:
  - Defines a single-layer linear model with 4 input features and 3 output features using custom weights.
  - Performs inference with input tensor `[1, 2, 3, 4]` and prints the result.
  - Exports the model to ONNX format with dynamic batch size support, saved to `models/example_dynamic_shape.onnx`.
- **Key Features**:
  - Explicit focus on dynamic batch size in ONNX export.
  - Simplified single-layer architecture.

## Requirements

- **Python**: `3.10.12`
- **Dependencies**: The required packages and versions are specified in `requirements.txt`:
  ```
  torch==2.7.0
  numpy==2.2.5
  onnx==1.18.0
  ```
- **Installation**: Install the dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### Validate ONNX Model:
   To verify the exported ONNX models, use the `onnx` library:
   ```python
   import onnx
   model = onnx.load("models/example.onnx")
   onnx.checker.check_model(model)
   ```

## File Structure

```
.
├── linear_model.py               # Single-layer linear model
├── linear_two_head_model.py      # Two-head linear model
├── linear_model_dynamic.py       # Single-layer model with dynamic shape
├── requirements.txt              # Dependency specifications
└── models/                       # Directory for exported ONNX models
    ├── example.onnx
    ├── example_two_head.onnx
    └── example_dynamic_shape.onnx
```

## Notes

- **Input Shape**: The scripts expect a 1D input tensor of shape `(4,)` for inference. The ONNX export uses a dummy input of shape `(1, 4)` to match this expectation.
- **Directory Creation**: Each script automatically creates the `models/` directory if it doesn't exist.
- **Dynamic Shapes**: All scripts support dynamic batch sizes in ONNX export via the `dynamic_axes` parameter.
- **Debugging**: The ONNX export uses `verbose=True` to print detailed information, which is useful for troubleshooting.
- **Limitations**: The scripts use hardcoded weights and inputs for simplicity. For production use, consider parameterizing these values or adding error handling.
