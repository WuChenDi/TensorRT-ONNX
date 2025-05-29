import onnx
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
        - 使用dict.get并显式转换为字符串，避免Pylance类型错误。
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

def main() -> None:
    """加载并解析ONNX模型，打印其输入、输出、节点和初始化权重信息。"""
    # 定义模型路径
    model_path = get_onnx_path(__file__, "sample-convnet.onnx")

    # 加载并验证模型
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到：{model_path}")
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"加载或验证模型失败：{str(e)}")
        return

    # 访问计算图组件
    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node
    inputs = graph.input
    outputs = graph.output

    # 解析输入信息
    print("\n" + "="*50)
    print("解析输入信息")
    print("="*50)
    for input in inputs:
        input_shape = [d.dim_value if d.dim_value != 0 else None for d in input.type.tensor_type.shape.dim]
        print(f"输入信息：\n"
              f"  名称：     {input.name}\n"
              f"  数据类型： {get_tensor_dtype(input.type.tensor_type.elem_type)}\n"
              f"  形状：     {input_shape}")

    # 解析输出信息
    print("\n" + "="*50)
    print("解析输出信息")
    print("="*50)
    for output in outputs:
        output_shape = [d.dim_value if d.dim_value != 0 else None for d in output.type.tensor_type.shape.dim]
        print(f"输出信息：\n"
              f"  名称：     {output.name}\n"
              f"  数据类型： {get_tensor_dtype(output.type.tensor_type.elem_type)}\n"
              f"  形状：     {output_shape}")

    # 解析节点信息
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

    # 解析初始化权重信息
    print("\n" + "="*50)
    print("解析初始化权重信息")
    print("="*50)
    for initializer in initializers:
        print(f"初始化权重信息：\n"
              f"  名称：     {initializer.name}\n"
              f"  数据类型： {get_tensor_dtype(initializer.data_type)}\n"
              f"  形状：     {list(initializer.dims)}")

if __name__ == "__main__":
    main()
