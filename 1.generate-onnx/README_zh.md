# PyTorch 线性模型脚本

本仓库包含三个使用 PyTorch 实现的简单线性模型脚本，用于执行推理并将模型导出为 ONNX 格式。每个脚本设计简洁，展示线性神经网络的不同配置。

## 脚本概览

### 1. `linear_model.py`
- **用途**: 实现单层线性模型，使用自定义权重，执行推理并导出为 ONNX。
- **功能**:
  - 定义一个单层线性模型，输入 4 个特征，输出 3 个特征，使用自定义权重。
  - 使用输入张量 `[1, 2, 3, 4]` 进行推理并打印结果。
  - 将模型导出为 ONNX 格式，支持动态批次大小，保存至 `models/example.onnx`。
- **主要特性**:
  - 单层线性模型，支持可选偏置（默认：禁用）。
  - ONNX 导出支持动态批次大小。

### 2. `linear_two_head_model.py`
- **用途**: 实现具有两个并行线性层的模型，执行推理并导出为 ONNX。
- **功能**:
  - 定义一个双头线性模型，包含两个并行的线性层，每层输入 4 个特征，输出 3 个特征，使用不同的自定义权重。
  - 使用输入张量 `[1, 2, 3, 4]` 进行推理，打印两个输出。
  - 将模型导出为 ONNX 格式，保存至 `models/example_two_head.onnx`。
- **主要特性**:
  - 两个并行线性层，分别使用独立权重。
  - ONNX 导出支持多个输出（`output0`, `output1`）。

### 3. `linear_model_dynamic.py`
- **用途**: 与 `linear_model.py` 类似，强调 ONNX 导出的动态形状支持。
- **功能**:
  - 定义一个单层线性模型，输入 4 个特征，输出 3 个特征，使用自定义权重。
  - 使用输入张量 `[1, 2, 3, 4]` 进行推理并打印结果。
  - 将模型导出为 ONNX 格式，支持动态批次大小，保存至 `models/example_dynamic_shape.onnx`。
- **主要特性**:
  - 明确支持 ONNX 导出的动态批次大小。
  - 简化的单层架构。

## 运行环境

- **Python**: `3.10.12`
- **依赖**: 所需包及版本在 `requirements.txt` 中指定：
  ```
  torch==2.7.0
  numpy==2.2.5
  onnx==1.18.0
  ```
- **安装**: 使用以下命令安装依赖：
  ```bash
  pip install -r requirements.txt
  ```

### 验证 ONNX 模型:
   使用 `onnx` 库验证导出的 ONNX 模型：
   ```python
   import onnx
   model = onnx.load("models/example.onnx")
   onnx.checker.check_model(model)
   ```

## 文件结构

```
.
├── linear_model.py               # 单层线性模型
├── linear_two_head_model.py      # 双头线性模型
├── linear_model_dynamic.py       # 单层线性模型（动态形状）
├── requirements.txt              # 依赖规格文件
└── models/                       # 导出的 ONNX 模型目录
    ├── example.onnx
    ├── example_two_head.onnx
    └── example_dynamic_shape.onnx
```

## 注意事项

- **输入形状**: 脚本推理时使用形状为 `(4,)` 的 1D 输入张量。ONNX 导出使用形状为 `(1, 4)` 的虚拟输入以匹配此期望。
- **目录创建**: 每个脚本会自动创建 `models/` 目录（如果不存在）。
- **动态形状**: 所有脚本通过 `dynamic_axes` 参数支持 ONNX 导出的动态批次大小。
- **调试**: ONNX 导出使用 `verbose=True` 打印详细信息，便于排查问题。
- **限制**: 脚本使用硬编码的权重和输入以保持简单。生产环境中，建议参数化这些值或添加错误处理。
