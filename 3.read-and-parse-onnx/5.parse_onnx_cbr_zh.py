import torch
import torch.nn as nn
import torch.onnx
import onnx
import numpy as np
import os
from onnx import TensorProto
from typing import Dict
from utils import get_onnx_path

def get_tensor_dtype(tensor_type: int) -> str:
    """将TensorProto数据类型映射为可读的字符串。

    参数:
        tensor_type: 表示TensorProto数据类型的整数（如1表示FLOAT）。

    返回:
        str: 可读的数据类型（如'FLOAT', 'INT32'）。

    注意:
        - 使用dict.get并显式转换为字符串，确保类型安全。
        - 未知类型返回'Unknown(tensor_type)'以提高清晰度。
    """
    dtype_map: Dict[int, str] = {
        TensorProto.FLOAT: 'FLOAT',
        TensorProto.INT32: 'INT32',
        TensorProto.INT64: 'INT64',
        TensorProto.DOUBLE: 'DOUBLE',
        TensorProto.UINT8: 'UINT8',
        TensorProto.INT8: 'INT8'
    }
    return dtype_map.get(tensor_type, f'Unknown({str(tensor_type)})')

def parse_onnx(model: onnx.ModelProto) -> None:
    """解析并打印ONNX模型结构。

    参数:
        model: 要解析的ONNX模型。

    注意:
        - 从model.graph解析输入、输出和节点。
        - 打印名称、数据类型、形状和节点属性。
    """
    graph = model.graph
    inputs = graph.input
    outputs = graph.output
    nodes = graph.node

    print("\n" + "="*50)
    print("解析输入信息")
    print("="*50)
    for input in inputs:
        input_shape = [d.dim_value if d.dim_value != 0 else None for d in input.type.tensor_type.shape.dim]
        print(f"输入信息：\n"
              f"  名称：     {input.name}\n"
              f"  数据类型： {get_tensor_dtype(input.type.tensor_type.elem_type)}\n"
              f"  形状：     {input_shape}")

    print("\n" + "="*50)
    print("解析输出信息")
    print("="*50)
    for output in outputs:
        output_shape = [d.dim_value if d.dim_value != 0 else None for d in output.type.tensor_type.shape.dim]
        print(f"输出信息：\n"
              f"  名称：     {output.name}\n"
              f"  数据类型： {get_tensor_dtype(output.type.tensor_type.elem_type)}\n"
              f"  形状：     {output_shape}")

    print("\n" + "="*50)
    print("解析节点信息")
    print("="*50)
    for node in nodes:
        attributes = [f"{attr.name}: {attr.ints or attr.floats or attr.s.decode()}" for attr in node.attribute]
        print(f"节点信息：\n"
              f"  名称：     {node.name}\n"
              f"  操作类型： {node.op_type}\n"
              f"  输入：     {node.input}\n"
              f"  输出：     {node.output}\n"
              f"  属性：     {attributes}")

def read_weight(initializer: onnx.TensorProto) -> None:
    """读取并打印ONNX初始化权重信息。

    参数:
        initializer: 包含权重数据的TensorProto对象。

    注意:
        - 打印初始化的名称、数据类型和形状。
    """
    shape = initializer.dims
    data  = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(shape)
    print("\n" + "="*50)
    print("解析权重数据")
    print("="*50)
    print(f"初始化权重信息：\n"
          f"  名称：     {initializer.name}\n"
          f"  数据类型： {get_tensor_dtype(initializer.data_type)}\n"
          f"  形状：     {list(shape)}\n"
          f"  数据：     \n{data}")

class Model(torch.nn.Module):
    """一个简单的卷积神经网络模型，包含Conv2d、BatchNorm2d和LeakyReLU。

    网络结构：
        input (1, 3, 5, 5)
          |
        Conv2d (in=3, out=16, kernel=3)
          |
        BatchNorm2d (num_features=16)
          |
        LeakyReLU
          |
        output (1, 16, 3, 3)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(num_features=16)
        self.act1  = nn.LeakyReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

def export_norm_onnx() -> None:
    """将PyTorch模型导出为ONNX格式。

    注意:
        - 导出一个CBR模型，输入形状为(1, 3, 5, 5)。
        - 保存至'./models/sample-cbr.onnx'，使用opset_version 15。
        - 输入名称：'input0'，输出名称：'output0'。
    """
    try:
        # 创建输入和模型
        input = torch.rand(1, 3, 5, 5)
        model = Model()
        model.eval()

        # 定义输出路径
        output_path = get_onnx_path(__file__, "sample-cbr.onnx")

        # 导出为ONNX
        torch.onnx.export(
            model         = model,
            args          = (input,),
            f             = output_path,
            input_names   = ["input0"],
            output_names  = ["output0"],
            opset_version = 15
        )
        print(f"成功导出ONNX模型至 {output_path}")
    except Exception as e:
        print(f"导出ONNX模型失败：{str(e)}")

def main() -> None:
    """主函数，用于导出和解析ONNX模型。"""
    # 导出模型为ONNX
    export_norm_onnx()

    # 加载并解析ONNX模型
    try:
        model_path = get_onnx_path(__file__, "sample-cbr.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到：{model_path}")
        
        model = onnx.load_model(model_path)
        onnx.checker.check_model(model)
        
        # 解析模型结构
        parse_onnx(model)

        # 解析初始化权重
        print("\n" + "="*50)
        print("解析初始化权重信息")
        print("="*50)
        initializers = model.graph.initializer
        if not initializers:
            print("模型中未找到初始化权重。")
        for item in initializers:
            read_weight(item)
            
    except Exception as e:
        print(f"加载或解析模型失败：{str(e)}")

if __name__ == "__main__":
    main()
