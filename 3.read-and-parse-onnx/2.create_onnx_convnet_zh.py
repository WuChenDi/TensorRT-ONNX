# ------------------------------------------------------------------------------
# 使用onnx.helper创建一个最基本的ConvNet
#         input (ch=3, h=64, w=64)
#           |
#          Conv (in_ch=3, out_ch=32, kernel=3, pads=1)
#           |
#        BatchNorm
#           |
#          ReLU
#           |
#         GlobalAvgPool
#           |
#          Conv (in_ch=32, out_ch=16, kernel=1, pads=0)
#           |
#         output (ch=16, h=1, w=1)
# ------------------------------------------------------------------------------

import numpy as np
import onnx
from onnx import helper, TensorProto
from typing import List
from numpy import ndarray
from utils import get_onnx_path

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: int = TensorProto.FLOAT
) -> onnx.TensorProto:
    """创建权重或偏置的初始化张量 (TensorProto)。

    参数:
        name: 张量的名称。
        tensor_array: 包含张量数据的NumPy数组。
        data_type: ONNX数据类型（默认：FLOAT）。

    返回:
        onnx.TensorProto: 初始化后的张量。
    """
    initializer = helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist()
    )
    return initializer

def infer_onnx(model: onnx.ModelProto) -> None:
    """使用ONNX Runtime在ONNX模型上进行推理以验证功能。

    参数:
        model: 要推理的ONNX模型。

    注意:
        - 需要onnxruntime和numpy包。通过pip安装：pip install onnxruntime numpy
        - 输入形状：(1, 3, 64, 64)；输出形状：(1, 16, 1, 1)。
        - 输出为NumPy数组列表，shape属性返回一个元组（如 (1, 16, 1, 1)）。
        - session.run 可能返回 Sequence[ndarray | SparseTensor | list | dict]，但本模型输出 ndarray。
        - 确保onnxruntime支持IR版本8（需要 >= 1.10.0）。
    """
    try:
        import onnxruntime as ort

        # 检查onnxruntime版本
        import pkg_resources
        ort_version = pkg_resources.get_distribution("onnxruntime").version
        if ort_version < "1.10.0":
            print(f"警告：onnxruntime {ort_version} 可能不支持IR版本 {model.ir_version}。建议升级到 >= 1.10.0。")

        # 创建随机输入数据
        input_data: ndarray = np.random.randn(1, 3, 64, 64).astype(np.float32)

        # 创建推理会话
        session = ort.InferenceSession(model.SerializeToString())
        
        # 执行推理
        inputs = {"input0": input_data}
        # 预期 List[ndarray]，以下验证
        outputs: List[ndarray] = session.run(None, inputs)  # type: ignore
        
        # 检查输出并验证类型
        if not outputs:
            raise ValueError("推理未产生输出。")
        if not isinstance(outputs[0], np.ndarray):
            raise TypeError(f"预期输出类型 np.ndarray，得到 {type(outputs[0])}")
        print("推理输出形状:", outputs[0].shape)  # 预期：(1, 16, 1, 1)

    except ImportError:
        print("未安装ONNX Runtime或NumPy，跳过推理。")
    except Exception as e:
        print(f"推理失败：{str(e)}。请检查onnxruntime版本（IR版本 {model.ir_version}）。")

def main() -> None:
    """创建并保存一个简单的ConvNet ONNX模型。"""
    # 设置随机种子以确保可复现性
    np.random.seed(42)

    # 定义模型维度
    input_shape = [1, 3, 64, 64]  # [批次, 通道, 高, 宽]
    output_shape = [1, 16, 1, 1]  # [批次, 通道, 高, 宽]

    # 创建输入和输出张量 (ValueInfoProto)
    input = helper.make_tensor_value_info("input0", TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_shape)

    # 创建第一个Conv节点
    conv1_out_ch = 32
    conv1_kernel = 3
    conv1_pads = 1
    conv1_weight = np.random.rand(conv1_out_ch, 3, conv1_kernel, conv1_kernel).astype(np.float32)
    conv1_bias = np.random.rand(conv1_out_ch).astype(np.float32)

    conv1_weight_initializer = create_initializer_tensor("conv2d_1.weight", conv1_weight)
    conv1_bias_initializer = create_initializer_tensor("conv2d_1.bias", conv1_bias)

    conv1_node = helper.make_node(
        name="conv2d_1",
        op_type="Conv",
        inputs=["input0", "conv2d_1.weight", "conv2d_1.bias"],
        outputs=["conv2d_1.output"],
        kernel_shape=[conv1_kernel, conv1_kernel],
        pads=[conv1_pads, conv1_pads, conv1_pads, conv1_pads],
    )

    # 创建BatchNorm节点
    bn1_scale = np.random.rand(conv1_out_ch).astype(np.float32)
    bn1_bias = np.random.rand(conv1_out_ch).astype(np.float32)
    bn1_mean = np.random.rand(conv1_out_ch).astype(np.float32)
    bn1_var = np.random.rand(conv1_out_ch).astype(np.float32)

    bn1_scale_initializer = create_initializer_tensor("batchNorm1.scale", bn1_scale)
    bn1_bias_initializer = create_initializer_tensor("batchNorm1.bias", bn1_bias)
    bn1_mean_initializer = create_initializer_tensor("batchNorm1.mean", bn1_mean)
    bn1_var_initializer = create_initializer_tensor("batchNorm1.var", bn1_var)

    bn1_node = helper.make_node(
        name="batchNorm1",
        op_type="BatchNormalization",
        inputs=[
            "conv2d_1.output",
            "batchNorm1.scale",
            "batchNorm1.bias",
            "batchNorm1.mean",
            "batchNorm1.var"
        ],
        outputs=["batchNorm1.output"],
    )

    # 创建ReLU节点
    relu1_node = helper.make_node(
        name="relu1",
        op_type="Relu",
        inputs=["batchNorm1.output"],
        outputs=["relu1.output"],
    )

    # 创建GlobalAveragePool节点
    global_avg_pool1_node = helper.make_node(
        name="global_avg_pool1",
        op_type="GlobalAveragePool",
        inputs=["relu1.output"],
        outputs=["global_avg_pool1.output"],
    )

    # 创建第二个Conv节点
    conv2_out_ch = 16
    conv2_kernel = 1
    conv2_pads = 0
    conv2_weight = np.random.rand(conv2_out_ch, conv1_out_ch, conv2_kernel, conv2_kernel).astype(np.float32)
    conv2_bias = np.random.rand(conv2_out_ch).astype(np.float32)

    conv2_weight_initializer = create_initializer_tensor("conv2d_2.weight", conv2_weight)
    conv2_bias_initializer = create_initializer_tensor("conv2d_2.bias", conv2_bias)

    conv2_node = helper.make_node(
        name="conv2d_2",
        op_type="Conv",
        inputs=["global_avg_pool1.output", "conv2d_2.weight", "conv2d_2.bias"],
        outputs=["output0"],
        kernel_shape=[conv2_kernel, conv2_kernel],
        pads=[conv2_pads, conv2_pads, conv2_pads, conv2_pads],
    )

    # 创建计算图 (GraphProto)
    graph = helper.make_graph(
        name="sample-convnet",
        inputs=[input],
        outputs=[output],
        nodes=[
            conv1_node,
            bn1_node,
            relu1_node,
            global_avg_pool1_node,
            conv2_node
        ],
        initializer=[
            conv1_weight_initializer,
            conv1_bias_initializer,
            bn1_scale_initializer,
            bn1_bias_initializer,
            bn1_mean_initializer,
            bn1_var_initializer,
            conv2_weight_initializer,
            conv2_bias_initializer
        ],
    )

    # 创建模型 (ModelProto)
    model = helper.make_model(
        graph,
        producer_name="onnx-sample",
        opset_imports=[helper.make_operatorsetid("", 15)]  # ONNX操作集版本15
    )
    model.ir_version = 8  # 设置IR版本以确保onnxruntime兼容性

    # 推断形状并验证模型
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    # 保存模型
    try:
        # 保存模型到文件
        output_path = get_onnx_path(__file__, "sample-convnet.onnx")
        onnx.save(model, output_path)  # 保存模型到文件
        print(f"恭喜！成功创建 {output_path}")
    except Exception as e:
        print(f"保存模型失败：{str(e)}")

    # 执行推理以验证
    infer_onnx(model)

if __name__ == "__main__":
    main()
