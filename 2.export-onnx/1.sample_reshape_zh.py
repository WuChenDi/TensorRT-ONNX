import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
from utils import get_onnx_path

class Model(nn.Module):
    """一个卷积神经网络，包含两个卷积块、自适应池化和全连接层。"""
    
    def __init__(self):
        """
        初始化模型，包含两个卷积块、自适应池化和全连接层。

        模型结构包括：
        - Conv2d (3->16, 3x3) + BatchNorm2d + ReLU
        - Conv2d (16->64, 5x5) + BatchNorm2d + ReLU
        - 自适应池化，压缩空间维度
        - 全连接层 (64->10) 用于分类
        """
        super().__init__()
        self.conv1   = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # 3x3卷积
        self.bn1     = nn.BatchNorm2d(num_features=16)  # 批归一化，针对16个通道
        self.act1    = nn.ReLU()  # ReLU激活函数
        self.conv2   = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, padding=2)  # 5x5卷积
        self.bn2     = nn.BatchNorm2d(num_features=64)  # 批归一化，针对64个通道
        self.act2    = nn.ReLU()  # ReLU激活函数
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应池化到1D
        self.head    = nn.Linear(in_features=64, out_features=10)  # 全连接层输出

    def forward(self, x):
        """
        前向传播，通过模型处理输入。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 3, height, width)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, 10)。
        """
        x = self.conv1(x)  # 应用第一层卷积
        x = self.bn1(x)    # 应用批归一化
        x = self.act1(x)   # 应用ReLU激活
        x = self.conv2(x)  # 应用第二层卷积
        x = self.bn2(x)    # 应用批归一化
        x = self.act2(x)   # 应用ReLU激活
        # 展平空间维度（H, W）为单一维度
        # 注意：torch.flatten(x, 2, 3) 在ONNX中会生成shape->slice->concat->reshape节点
        x = torch.flatten(x, 2, 3)  # (B, C, H, W) -> (B, C, H*W)
        x = self.avgpool(x)         # (B, C, H*W) -> (B, C, 1)
        x = torch.flatten(x, 1)     # (B, C, 1) -> (B, C)
        x = self.head(x)            # (B, C) -> (B, 10)
        return x

def infer():
    """使用模型进行推理，输入随机张量。"""
    # 定义输入张量（batch_size=1, 通道=3, 高=64, 宽=64）
    input_tensor = torch.rand(1, 3, 64, 64)
    
    # 初始化模型
    model = Model()
    
    # 执行推理并打印输出形状
    output = model(input_tensor)
    print("推理输出形状:", output.shape)

def export_norm_onnx():
    """将模型导出为ONNX格式并进行验证和简化。"""
    # 定义虚拟输入张量（batch_size, 通道, 高, 宽），用于ONNX导出
    input_tensor = torch.rand(1, 3, 64, 64)
    
    # 初始化模型并设置为评估模式
    model = Model()
    model.eval()  # 切换到评估模式以确保导出稳定

    # 定义ONNX模型的保存路径
    output_path = get_onnx_path(__file__, "sample-reshape.onnx")

    # 导出模型到ONNX格式
    # 注意：torch.flatten会生成shape->slice->concat->reshape节点。
    # 使用onnx-simplifier可将这些节点合并为单个Flatten或Reshape节点。
    # BatchNorm层在do_constant_folding=True时可能被折叠到Conv层。
    torch.onnx.export(
        model=model,              # 要导出的模型
        args=(input_tensor,),     # 虚拟输入，用于跟踪计算图
        f=output_path,            # 输出文件路径
        input_names=["input0"],   # 输入张量名称
        output_names=["output0"], # 输出张量名称
        opset_version=15,         # ONNX操作集版本，确保兼容性
        dynamic_axes={
            'input0': {0: 'batch'},   # 支持动态批次大小
            'output0': {0: 'batch'}   # 支持动态批次大小
        },
        do_constant_folding=True, # 优化：折叠常量（如BatchNorm到Conv）
        verbose=True              # 打印导出详情，便于调试
    )
    
    # 加载并验证ONNX模型
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)  # 检查模型完整性
    
    # 使用onnx-simplifier简化ONNX模型
    # 这可以将复杂的节点（如shape->slice->concat->reshape）合并为更简单的节点
    print(f"使用onnx-simplifier {onnxsim.__version__} 进行简化...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "ONNX简化失败"
    onnx.save(model_onnx, output_path)  # 保存简化后的模型
    print(f"ONNX导出和简化完成，模型保存至: {output_path}")

if __name__ == "__main__":
    infer()            # 执行推理
    export_norm_onnx() # 导出模型到ONNX
