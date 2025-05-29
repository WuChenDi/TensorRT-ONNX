# 理解onnx中的组织结构
# - ModelProto: 描述的是整个模型的信息
#   - GraphProto: 描述的是整个网络的信息
#     - NodeProto: 描述的是各个计算节点，比如conv, linear
#     - TensorProto: 描述的是tensor的信息，主要包括权重
#     - ValueInfoProto: 描述的是input/output信息
#     - AttributeProto: 描述的是node节点的各种属性信息
# ------------------------------------------------------------------------------

import onnx
from onnx import helper
from onnx import TensorProto
from utils import get_onnx_path

def create_onnx():
    """创建一个简单的ONNX模型，计算 y = a * x + b。

    返回:
        onnx.ModelProto: 构建的ONNX模型。
    """
    # 创建输入和输出张量 (ValueInfoProto)
    # 每个张量定义了名称、类型 (FLOAT) 和形状 [10, 10]
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])  # 输入张量 a
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])  # 输入张量 x
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])  # 输入张量 b
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 10])  # 输出张量 y

    # 创建计算节点 (NodeProto)
    # 乘法节点：c = a * x
    mul = helper.make_node(
        op_type='Mul',          # 操作类型
        inputs=['a', 'x'],      # 输入张量
        outputs=['c'],          # 输出张量
        name='multiply'         # 节点名称，便于调试
    )
    # 加法节点：y = c + b
    add = helper.make_node(
        op_type='Add',          # 操作类型
        inputs=['c', 'b'],      # 输入张量
        outputs=['y'],          # 输出张量
        name='add'              # 节点名称，便于调试
    )

    # 创建计算图 (GraphProto)
    # 计算图包括节点、名称、输入和输出
    graph = helper.make_graph(
        nodes=[mul, add],       # 节点列表
        name='sample-linear',   # 计算图名称
        inputs=[a, x, b],       # 输入张量
        outputs=[y]             # 输出张量
    )

    # 创建模型 (ModelProto)
    # 指定操作集版本和IR版本以确保兼容性
    model = helper.make_model(
        graph,
        producer_name='onnx_example',
        opset_imports=[helper.make_operatorsetid('', 15)]  # ONNX操作集版本15
    )
    model.ir_version = 8  # 设置IR版本为8，确保与大多数onnxruntime版本兼容

    # 验证模型
    onnx.checker.check_model(model)  # 确保模型格式正确

    # 保存模型到文件
    output_path = get_onnx_path(__file__, "sample-linear.onnx")
    onnx.save(model, output_path)  # 保存模型到文件
    print(f"模型保存至: {output_path}")

    return model

if __name__ == "__main__":
    model = create_onnx()  # 创建并保存ONNX模型
