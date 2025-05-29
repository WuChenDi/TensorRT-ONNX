# 卷积模型 PyTorch 脚本

本仓库包含使用 PyTorch 实现的卷积神经网络（CNN）模型脚本，支持推理和 ONNX 格式导出。这些脚本展示了不同的 CNN 架构和 ONNX 导出技术，包括动态形状支持和模型简化。

## 脚本概览

### 1. `sample_reshape.py`
- **用途**：实现一个包含两个卷积块、自适应池化和线性头的 CNN 模型，并导出为 ONNX 格式，同时进行模型简化。
- **功能**：
  - 定义一个模型，包含两个卷积块（`Conv2d`、`BatchNorm2d`、`ReLU`）、自适应池化以及一个线性层（64→10 输出）。
  - 使用形状为 `(1, 3, 64, 64)` 的随机输入张量进行推理，并打印输出形状。
  - 将模型导出为 ONNX 格式，支持动态批次大小，保存至 `models/sample-reshape.onnx`。
  - 使用 `onnx-simplifier` 简化 ONNX 模型，优化节点（例如，将 `shape→slice→concat→reshape` 合并为单个 `Flatten` 或 `Reshape` 节点）。
- **主要特性**：
  - 两个卷积块，包含批归一化和 ReLU 激活。
  - 自适应池化支持可变输入尺寸。
  - ONNX 导出时启用常量折叠（例如，将 `BatchNorm2d` 融合到 `Conv2d`）。
  - 模型简化以减少图复杂性。

### 2. `sample_cbr.py`
- **用途**：实现一个简单的卷积块（`Conv2d`、`BatchNorm2d`、`ReLU`）并导出为 ONNX 格式。
- **功能**：
  - 定义一个卷积块，输入 3 个通道，输出 16 个通道，使用 3x3 卷积核。
  - 使用形状为 `(1, 3, 5, 5)` 的随机输入张量进行推理，并打印输出形状。
  - 将模型导出为 ONNX 格式，支持动态批次大小，保存至 `models/sample-cbr.onnx`。
- **主要特性**：
  - 简单的卷积块，包含批归一化和 ReLU 激活。
  - ONNX 导出时启用常量折叠，可能将 `BatchNorm2d` 融合到 `Conv2d`。

### 3. `load_torchvision.py`
- **用途**：将预训练的 torchvision 模型（例如 ResNet、VGG、MobileNet）导出为 ONNX 格式，并进行简化。
- **功能**：
  - 支持多种预训练模型：ResNet50、VGG11、MobileNetV3-Small、EfficientNet-B0、EfficientNetV2-S 和 RegNet-X-1.6GF。
  - 使用形状为 `(1, 3, 224, 224)` 的随机输入张量在 CUDA 设备上进行推理，并打印输出形状。
  - 将选定的模型导出为 ONNX 格式，支持动态批次大小，保存至 `models/<model_name>.onnx`（例如 `resnet50.onnx`）。
  - 简化 ONNX 模型，移除冗余节点（例如，残差连接中的 `Identity` 节点）。
- **主要特性**：
  - 支持多种预训练 torchvision 模型。
  - 需要 CUDA 设备支持。
  - ONNX 导出时启用常量折叠和模型简化。

### 辅助脚本：`utils.py`
- **用途**：提供管理 ONNX 文件路径和扩展名的辅助函数。
- **功能**：
  - `ensure_extension`：确保文件具有 `.onnx` 扩展名。
  - `get_onnx_path`：生成标准化的 ONNX 模型保存路径，必要时创建 `models/` 目录。

## 依赖要求

- **Python**：`3.10.12`
- **依赖包**：安装 `requirements.txt` 中列出的包：
  ```
  torch==2.7.0
  numpy==2.2.5
  onnx==1.18.0
  onnxsim==0.4.36
  torchvision==0.22.0
  ```
- **安装**：
  ```bash
  pip install -r requirements.txt
  ```
- **注意**：`load_torchvision.py` 需要 CUDA 设备支持。

### 验证 ONNX 模型：
使用 `onnx` 库验证导出的 ONNX 模型：
```python
import onnx
model = onnx.load("models/sample-reshape.onnx")
onnx.checker.check_model(model)
```

## 文件结构

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

## 使用方法

1. **运行 `sample_reshape.py`**：
   ```bash
   python sample_reshape.py
   ```
   执行推理并将模型导出至 `models/sample-reshape.onnx`。

2. **运行 `sample_cbr.py`**：
   ```bash
   python sample_cbr.py
   ```
   执行推理并将模型导出至 `models/sample-cbr.onnx`。

3. **运行 `load_torchvision.py`**：
   指定模型类型（例如 `resnet`、`vgg`）和输出目录：
   ```bash
   python load_torchvision.py -t resnet -d models/
   ```
   执行推理并将指定模型（例如 ResNet50）导出至 `models/resnet50.onnx`。需要 CUDA 设备支持。

## 注意事项

- **输入形状**：
  - `sample_reshape.py`：期望输入张量形状为 `(batch_size, 3, 64, 64)`。
  - `sample_cbr.py`：期望输入张量形状为 `(batch_size, 3, 5, 5)`。
  - `load_torchvision.py`：期望输入张量形状为 `(batch_size, 3, 224, 224)`。
- **动态形状**：所有脚本通过 `dynamic_axes` 参数支持 ONNX 导出的动态批次大小。
- **ONNX 优化**：
  - `do_constant_folding=True` 可能将 `BatchNorm2d` 融合到 `Conv2d` 层。
  - `sample_reshape.py` 和 `load_torchvision.py` 使用 `onnx-simplifier` 优化 ONNX 图（例如，合并复杂节点或移除 `Identity` 节点）。
- **调试**：在 `torch.onnx.export` 中使用 `verbose=True` 获取详细导出日志。使用 Netron 等工具检查 ONNX 图，分析节点（如 ResNet 中的 `Identity` 节点）。
- **限制**：
  - `load_torchvision.py` 需要 CUDA 设备。
  - 为简化起见，使用硬编码的输入尺寸。生产环境中建议参数化输入或添加错误处理。
