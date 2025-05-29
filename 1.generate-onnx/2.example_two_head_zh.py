import torch
import torch.nn as nn
import torch.onnx
from utils import get_onnx_path

class Model(torch.nn.Module):
    """一个具有两个并行线性层的模型，每个层使用自定义权重。"""
    
    def __init__(self, in_features, out_features, weights1, weights2, bias=False):
        """
        初始化模型，包含两个线性层和自定义权重。

        参数:
            in_features (int): 输入特征数。
            out_features (int): 每个线性层的输出特征数。
            weights1 (torch.Tensor): 第一个线性层的权重，形状为 (out_features, in_features)。
            weights2 (torch.Tensor): 第二个线性层的权重，形状为 (out_features, in_features)。
            bias (bool): 是否包含偏置项，默认为 False。
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features, bias)  # 第一个线性层
        self.linear2 = nn.Linear(in_features, out_features, bias)  # 第二个线性层
        with torch.no_grad():
            self.linear1.weight.copy_(weights1)  # 设置第一个线性层的权重
            self.linear2.weight.copy_(weights2)  # 设置第二个线性层的权重

    def forward(self, x):
        """
        前向传播，通过两个线性层处理输入。

        参数:
            x (torch.Tensor): 输入张量，形状为 (in_features,)。

        返回:
            tuple: 两个线性层的输出，每个形状为 (out_features,)。
        """
        x1 = self.linear1(x)  # 第一个线性层的输出
        x2 = self.linear2(x)  # 第二个线性层的输出
        return x1, x2

def infer(weights1, weights2):
    """使用双头线性模型进行推理。"""
    # 定义输入张量（1D，4个特征）
    in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    
    # 初始化模型：4个输入特征，每个线性层3个输出特征
    model = Model(4, 3, weights1, weights2)
    
    # 执行推理并打印结果
    x1, x2 = model(in_features)
    print("推理结果：")
    print("输出 1:", x1)
    print("输出 2:", x2)

def export_onnx(weights1, weights2):
    """将双头线性模型导出为 ONNX 格式。"""
    # 定义虚拟输入张量（batch_size, in_features），用于 ONNX 导出
    input = torch.zeros(1, 4)  # 修正形状以匹配模型输入
    
    # 初始化模型并设置为评估模式
    model = Model(4, 3, weights1, weights2)
    model.eval()  # 切换到评估模式以确保导出稳定

    # 定义 ONNX 模型的保存路径
    output_path = get_onnx_path(__file__, "example_two_head.onnx")

    # 导出模型到 ONNX 格式
    torch.onnx.export(
        model=model,              # 要导出的模型
        args=(input,),            # 虚拟输入，用于跟踪计算图
        f=output_path,            # 输出文件路径
        input_names=["input0"],   # 输入张量名称
        output_names=["output0", "output1"],  # 输出张量名称
        opset_version=12,         # ONNX 操作集版本（12 确保兼容性）
        dynamic_axes={
            'input0': {0: 'batch_size'},   # 支持动态批次大小
            'output0': {0: 'batch_size'},  # 支持动态批次大小
            'output1': {0: 'batch_size'}   # 支持动态批次大小
        },
        do_constant_folding=True,  # 优化：折叠常量
        verbose=True               # 打印导出详情，便于调试
    )
    print(f"ONNX 导出完成，模型保存至: {output_path}")

if __name__ == "__main__":
    # 定义权重矩阵（3个输出特征 x 4个输入特征）
    weights1 = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ], dtype=torch.float32)
    weights2 = torch.tensor([
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ], dtype=torch.float32)
    
    infer(weights1, weights2)  # 执行推理
    export_onnx(weights1, weights2)  # 导出模型到 ONNX
