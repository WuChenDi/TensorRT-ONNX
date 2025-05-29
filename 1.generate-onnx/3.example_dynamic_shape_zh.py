import torch
import torch.nn as nn
import torch.onnx
from utils import get_onnx_path

class Model(torch.nn.Module):
    """一个简单的线性模型，支持自定义权重和可选偏置。"""
    
    def __init__(self, in_features, out_features, weights, bias=False):
        """
        初始化线性模型，设置自定义权重。

        参数:
            in_features (int): 输入特征数。
            out_features (int): 输出特征数。
            weights (torch.Tensor): 权重张量，形状为 (out_features, in_features)。
            bias (bool): 是否包含偏置项，默认为 False。
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)  # 创建线性层
        with torch.no_grad():
            self.linear.weight.copy_(weights)  # 设置自定义权重

    def forward(self, x):
        """
        前向传播，通过线性层处理输入。

        参数:
            x (torch.Tensor): 输入张量，形状为 (in_features,)。

        返回:
            torch.Tensor: 输出张量，形状为 (out_features,)。
        """
        x = self.linear(x)  # 应用线性变换
        return x

def infer(weights):
    """使用线性模型进行推理，基于预定义的输入和权重。"""
    # 定义输入张量（1D，4个特征）
    in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    
    # 初始化模型：4个输入特征，3个输出特征
    model = Model(4, 3, weights)
    
    # 执行推理并打印结果
    x = model(in_features)
    print("输入 [1, 2, 3, 4] 的推理结果为:", x)

def export_onnx(weights):
    """将线性模型导出为 ONNX 格式，支持动态批次大小。"""
    # 定义虚拟输入张量（batch_size, in_features），用于 ONNX 导出
    input = torch.zeros(1, 4)  # 修正形状以匹配模型输入
    
    # 初始化模型并设置为评估模式
    model = Model(4, 3, weights)
    model.eval()  # 切换到评估模式以确保导出稳定

    # 定义 ONNX 模型的保存路径
    output_path = get_onnx_path(__file__, "example_dynamic_shape.onnx")

    # 导出模型到 ONNX 格式
    torch.onnx.export(
        model=model,              # 要导出的模型
        args=(input,),            # 虚拟输入，用于跟踪计算图
        f=output_path,            # 输出文件路径
        input_names=["input0"],   # 输入张量名称
        output_names=["output0"], # 输出张量名称
        dynamic_axes={
            'input0': {0: 'batch'},   # 支持动态批次大小
            'output0': {0: 'batch'}   # 支持动态批次大小
        },
        opset_version=12,         # ONNX 操作集版本（12 确保兼容性）
        do_constant_folding=True, # 优化：折叠常量
        verbose=True              # 打印导出详情，便于调试
    )
    print(f"ONNX 导出完成，模型保存至: {output_path}")

if __name__ == "__main__":
    # 定义权重矩阵（3个输出特征 x 4个输入特征）
    weights = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ], dtype=torch.float32)
    
    infer(weights)      # 执行推理
    export_onnx(weights) # 导出模型到 ONNX
