# Convolutional Model PyTorch Scripts

This repository contains Python scripts that implement convolutional neural network (CNN) models using PyTorch, perform inference, and export the models to ONNX format. The scripts demonstrate various CNN architectures and ONNX export techniques, including support for dynamic shapes and model simplification.

## Scripts Overview

### 1. `sample_reshape.py`
- **Purpose**: Implements a CNN with two convolutional blocks, adaptive pooling, and a linear head, then exports it to ONNX with simplification.
- **Functionality**:
  - Defines a model with two convolutional blocks (`Conv2d`, `BatchNorm2d`, `ReLU`), adaptive pooling, and a linear layer (64→10 outputs).
  - Performs inference with a random input tensor of shape `(1, 3, 64, 64)` and prints the output shape.
  - Exports the model to ONNX format with dynamic batch size support, saved to `models/sample-reshape.onnx`.
  - Simplifies the ONNX model using `onnx-simplifier` to optimize nodes (e.g., merging `shape→slice→concat→reshape` into a single `Flatten` or `Reshape` node).
- **Key Features**:
  - Two convolutional blocks with batch normalization and ReLU activation.
  - Adaptive pooling to handle variable input sizes.
  - ONNX export with constant folding (e.g., fusing `BatchNorm2d` into `Conv2d`).
  - Model simplification to reduce graph complexity.

### 2. `sample_cbr.py`
- **Purpose**: Implements a simple convolutional block (`Conv2d`, `BatchNorm2d`, `ReLU`) and exports it to ONNX.
- **Functionality**:
  - Defines a single convolutional block with 3 input channels, 16 output channels, and a 3x3 kernel.
  - Performs inference with a random input tensor of shape `(1, 3, 5, 5)` and prints the output shape.
  - Exports the model to ONNX format with dynamic batch size support, saved to `models/sample-cbr.onnx`.
- **Key Features**:
  - Simple convolutional block with batch normalization and ReLU.
  - ONNX export with constant folding, potentially fusing `BatchNorm2d` into `Conv2d`.

### 3. `load_torchvision.py`
- **Purpose**: Exports pre-trained torchvision models (e.g., ResNet, VGG, MobileNet) to ONNX format with simplification.
- **Functionality**:
  - Supports multiple pre-trained models: ResNet50, VGG11, MobileNetV3-Small, EfficientNet-B0, EfficientNetV2-S, and RegNet-X-1.6GF.
  - Performs inference with a random input tensor of shape `(1, 3, 224, 224)` on a CUDA device and prints the output shape.
  - Exports the selected model to ONNX format with dynamic batch size support, saved to `models/<model_name>.onnx` (e.g., `resnet50.onnx`).
  - Simplifies the ONNX model to remove redundant nodes (e.g., `Identity` nodes from residual connections).
- **Key Features**:
  - Supports multiple pre-trained torchvision models.
  - Requires a CUDA-enabled GPU for execution.
  - ONNX export with constant folding and model simplification.

### Utility Script: `utils.py`
- **Purpose**: Provides helper functions for managing ONNX file paths and extensions.
- **Functionality**:
  - `ensure_extension`: Ensures a file has the `.onnx` extension.
  - `get_onnx_path`: Generates a standardized path for saving ONNX models, creating the `models/` directory if needed.

## Requirements

- **Python**: `3.10.12`
- **Dependencies**: Install the required packages listed in `requirements.txt`:
  ```
  torch==2.7.0
  numpy==2.2.5
  onnx==1.18.0
  onnxsim==0.4.36
  torchvision==0.22.0
  ```
- **Installation**:
  ```bash
  pip install -r requirements.txt
  ```
- **Note**: `load_torchvision.py` requires a CUDA-enabled GPU.

### Validate ONNX Model:
To verify the exported ONNX models, use the `onnx` library:
```python
import onnx
model = onnx.load("models/sample-reshape.onnx")
onnx.checker.check_model(model)
```

## File Structure

```
├── 2.export-onnx/
│   ├── 1.sample_reshape.py
│   ├── 2.sample_cbr.py
│   ├── 3.load_torchvision.py
│   ├── README.md
│   ├── models/
│   │   ├── resnet50.onnx
│   │   ├── sample-cbr.onnx
│   │   └── sample-reshape.onnx
│   └── utils.py
```

## Usage

1. **Run `sample_reshape.py`**:
   ```bash
   python sample_reshape.py
   ```
   This performs inference and exports the model to `models/sample-reshape.onnx`.

2. **Run `sample_cbr.py`**:
   ```bash
   python sample_cbr.py
   ```
   This performs inference and exports the model to `models/sample-cbr.onnx`.

3. **Run `load_torchvision.py`**:
   Specify a model type (e.g., `resnet`, `vgg`) and output directory:
   ```bash
   python load_torchvision.py -t resnet -d models/
   ```
   This performs inference and exports the specified model (e.g., ResNet50) to `models/resnet50.onnx`. Requires a CUDA-enabled GPU.

## Notes

- **Input Shapes**:
  - `sample_reshape.py`: Expects input tensor of shape `(batch_size, 3, 64, 64)`.
  - `sample_cbr.py`: Expects input tensor of shape `(batch_size, 3, 5, 5)`.
  - `load_torchvision.py`: Expects input tensor of shape `(batch_size, 3, 224, 224)`.
- **Dynamic Shapes**: All scripts support dynamic batch sizes in ONNX export via the `dynamic_axes` parameter.
- **ONNX Optimization**:
  - `do_constant_folding=True` may fuse `BatchNorm2d` into `Conv2d` layers.
  - `sample_reshape.py` and `load_torchvision.py` use `onnx-simplifier` to optimize the ONNX graph (e.g., merging complex nodes or removing `Identity` nodes).
- **Debugging**: Use `verbose=True` in `torch.onnx.export` for detailed export logs. Inspect ONNX graphs with tools like Netron to analyze nodes (e.g., `Identity` nodes in ResNet).
- **Limitations**:
  - `load_torchvision.py` requires a CUDA-enabled GPU.
  - Hardcoded input sizes are used for simplicity. For production, consider parameterizing inputs or adding error handling.
