import torch
import torch.nn as nn
import torch.onnx
from utils import get_onnx_path

class Model(nn.Module):
    """一个简单的卷积块，包含Conv2d、BatchNorm2d和ReLU激活函数。"""
    
    def __init__(self):
        """
        初始化卷积块。

        模型结构包括：
        - Conv2d：3个输入通道，16个输出通道，3x3卷积核
        - BatchNorm2d：对16个输出通道进行归一化
        - ReLU：应用非线性激活函数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)  # 3x3卷积层
        self.bn1   = nn.BatchNorm2d(num_features=16)  # 批归一化，针对16个通道
        self.act1  = nn.ReLU()  # ReLU激活函数

    def forward(self, x):
        """
        前向传播，通过卷积块处理输入。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 3, height, width)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, 16, height-2, width-2)。
        """
        x = self.conv1(x)  # 应用卷积操作
        x = self.bn1(x)    # 应用批归一化
        x = self.act1(x)   # 应用ReLU激活
        return x

def infer():
    """使用卷积模型进行推理，输入随机张量。"""
    # 定义输入张量（batch_size=1, 通道=3, 高=5, 宽=5）
    input_tensor = torch.rand(1, 3, 5, 5)
    
    # 初始化模型
    model = Model()
    
    # 执行推理并打印输出形状
    output = model(input_tensor)
    print("推理输出形状:", output.shape)

def export_norm_onnx():
    """将卷积模型导出为ONNX格式。"""
    # 定义虚拟输入张量（batch_size, 通道, 高, 宽），用于ONNX导出
    input_tensor = torch.rand(1, 3, 5, 5)
    
    # 初始化模型并设置为评估模式
    model = Model()
    model.eval()  # 切换到评估模式以确保导出稳定

    # 定义ONNX模型的保存路径
    output_path = get_onnx_path(__file__, "sample-cbr.onnx")

    # 导出模型到ONNX格式
    # 注意：在导出时，若设置do_constant_folding=True且模型处于eval模式，
    # BatchNorm2d可能会与Conv2d融合为单个Conv算子，以优化计算图。
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
    print(f"ONNX导出完成，模型保存至: {output_path}")

if __name__ == "__main__":
    infer()            # 执行推理
    export_norm_onnx() # 导出模型到ONNX
