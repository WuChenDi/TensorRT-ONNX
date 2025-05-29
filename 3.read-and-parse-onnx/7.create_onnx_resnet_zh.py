# 创建 ResNet-18 的 ONNX 模型
# - 包含初始卷积、最大池化、4 个残差阶段、全局平均池化和全连接层
# - 使用 opset_version=12，确保兼容性
# - 输出模型保存为 models/resnet18.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from utils import get_onnx_path

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: int = TensorProto.FLOAT
) -> onnx.TensorProto:
    """创建ONNX TensorProto初始化张量。"""
    if tensor_array.size == 0:
        raise ValueError(f"初始化器 {name} 为空")
    initializer = helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())
    return initializer

def parse_onnx(model: onnx.ModelProto) -> None:
    """解析并打印ONNX模型结构。"""
    try:
        graph = model.graph
        if not graph:
            raise ValueError("模型计算图为空")

        print(f"\n{'='*50}\n解析输入信息\n{'='*50}")
        for input in graph.input:
            input_shape = [d.dim_value if d.dim_value != 0 else None for d in input.type.tensor_type.shape.dim]
            print(f"输入信息:\n  名称:     {input.name}\n  数据类型: {TensorProto.DataType.Name(input.type.tensor_type.elem_type)}\n  形状:     {input_shape}")

        print(f"\n{'='*50}\n解析输出信息\n{'='*50}")
        for output in graph.output:
            output_shape = [d.dim_value if d.dim_value != 0 else None for d in output.type.tensor_type.shape.dim]
            print(f"输出信息:\n  名称:     {output.name}\n  数据类型: {TensorProto.DataType.Name(output.type.tensor_type.elem_type)}\n  形状:     {output_shape}")

        print(f"\n{'='*50}\n解析节点信息\n{'='*50}")
        for node in graph.node:
            attributes = [f"{attr.name}: {attr.ints if attr.type == onnx.AttributeProto.INTS else attr.floats if attr.type == onnx.AttributeProto.FLOATS else attr.f if attr.type == onnx.AttributeProto.FLOAT else attr.s.decode() if attr.type == onnx.AttributeProto.STRING else '不支持'}" for attr in node.attribute]
            print(f"节点信息:\n  名称:     {node.name}\n  操作类型: {node.op_type}\n  输入:     {node.input}\n  输出:     {node.output}\n  属性:     {attributes}")

        print(f"\n{'='*50}\n解析初始化权重信息\n{'='*50}")
        for initializer in graph.initializer:
            print(f"初始化权重信息:\n  名称:     {initializer.name}\n  数据类型: {TensorProto.DataType.Name(initializer.data_type)}\n  形状:     {list(initializer.dims)}")
    except Exception as e:
        print(f"解析模型失败: {str(e)}")

def conv_bn_relu(input_name, conv_weight_name, conv_bias_name, bn_scale_name, bn_bias_name,
                 bn_mean_name, bn_var_name, output_name, nodes, initializers,
                 conv_params, layer_name):
    """创建 Conv -> BatchNorm -> ReLU 的组合。"""
    conv_output = f"{layer_name}_conv_output"
    bn_output = f"{layer_name}_bn_output"
    relu_output = output_name

    # 卷积权重和偏置（Xavier 初始化）
    fan_in = conv_params['weight_shape'][1] * conv_params['kernel_shape'][0] * conv_params['kernel_shape'][1]
    fan_out = conv_params['weight_shape'][0]
    conv_weight = np.random.randn(*conv_params['weight_shape']).astype(np.float32) * np.sqrt(2.0 / (fan_in + fan_out))
    conv_bias = np.zeros((conv_params['weight_shape'][0],), dtype=np.float32)

    conv_weight_initializer = create_initializer_tensor(name=conv_weight_name, tensor_array=conv_weight)
    conv_bias_initializer = create_initializer_tensor(name=conv_bias_name, tensor_array=conv_bias)
    initializers.extend([conv_weight_initializer, conv_bias_initializer])

    # 卷积节点
    conv_node = helper.make_node(
        name=f"{layer_name}_conv",
        op_type="Conv",
        inputs=[input_name, conv_weight_name, conv_bias_name],
        outputs=[conv_output],
        kernel_shape=conv_params['kernel_shape'],
        pads=conv_params['pads'],
        strides=conv_params['strides']
    )
    nodes.append(conv_node)
    print(f"{layer_name} conv: input={input_name}, weight={conv_params['weight_shape']}, output={conv_output}")

    # BatchNorm 参数
    num_features = conv_params['weight_shape'][0]
    bn_scale = np.ones((num_features,), dtype=np.float32)
    bn_bias = np.zeros((num_features,), dtype=np.float32)
    bn_mean = np.zeros((num_features,), dtype=np.float32)
    bn_var = np.ones((num_features,), dtype=np.float32)

    bn_scale_initializer = create_initializer_tensor(name=bn_scale_name, tensor_array=bn_scale)
    bn_bias_initializer = create_initializer_tensor(name=bn_bias_name, tensor_array=bn_bias)
    bn_mean_initializer = create_initializer_tensor(name=bn_mean_name, tensor_array=bn_mean)
    bn_var_initializer = create_initializer_tensor(name=bn_var_name, tensor_array=bn_var)
    initializers.extend([bn_scale_initializer, bn_bias_initializer, bn_mean_initializer, bn_var_initializer])

    # BatchNorm 节点
    bn_node = helper.make_node(
        name=f"{layer_name}_bn",
        op_type="BatchNormalization",
        inputs=[conv_output, bn_scale_name, bn_bias_name, bn_mean_name, bn_var_name],
        outputs=[bn_output],
        epsilon=1e-5,
        momentum=0.9
    )
    nodes.append(bn_node)

    # ReLU 节点
    relu_node = helper.make_node(
        name=f"{layer_name}_relu",
        op_type="Relu",
        inputs=[bn_output],
        outputs=[relu_output]
    )
    nodes.append(relu_node)

    return relu_output

def basic_block(input_name, output_name, nodes, initializers, in_channels, out_channels,
                stride, layer_name, downsample=False):
    """创建 ResNet 的基本残差块（BasicBlock）。"""
    # 第一个卷积层
    conv1_weight_name = f"{layer_name}_conv1_weight"
    conv1_bias_name = f"{layer_name}_conv1_bias"
    bn1_scale_name = f"{layer_name}_bn1_scale"
    bn1_bias_name = f"{layer_name}_bn1_bias"
    bn1_mean_name = f"{layer_name}_bn1_mean"
    bn1_var_name = f"{layer_name}_bn1_var"

    conv1_params = {
        'weight_shape': [out_channels, in_channels, 3, 3],
        'kernel_shape': [3, 3],
        'pads': [1, 1, 1, 1],
        'strides': [stride, stride]
    }

    relu1_output = f"{layer_name}_relu1_output"
    relu1_output = conv_bn_relu(
        input_name=input_name,
        conv_weight_name=conv1_weight_name,
        conv_bias_name=conv1_bias_name,
        bn_scale_name=bn1_scale_name,
        bn_bias_name=bn1_bias_name,
        bn_mean_name=bn1_mean_name,
        bn_var_name=bn1_var_name,
        output_name=relu1_output,
        nodes=nodes,
        initializers=initializers,
        conv_params=conv1_params,
        layer_name=f"{layer_name}_conv1"
    )

    # 第二个卷积层
    conv2_weight_name = f"{layer_name}_conv2_weight"
    conv2_bias_name = f"{layer_name}_conv2_bias"
    bn2_scale_name = f"{layer_name}_bn2_scale"
    bn2_bias_name = f"{layer_name}_bn2_bias"
    bn2_mean_name = f"{layer_name}_bn2_mean"
    bn2_var_name = f"{layer_name}_bn2_var"

    conv2_params = {
        'weight_shape': [out_channels, out_channels, 3, 3],
        'kernel_shape': [3, 3],
        'pads': [1, 1, 1, 1],
        'strides': [1, 1]
    }

    conv2_output = f"{layer_name}_conv2_output"
    conv2_output = conv_bn_relu(
        input_name=relu1_output,
        conv_weight_name=conv2_weight_name,
        conv_bias_name=conv2_bias_name,
        bn_scale_name=bn2_scale_name,
        bn_bias_name=bn2_bias_name,
        bn_mean_name=bn2_mean_name,
        bn_var_name=bn2_var_name,
        output_name=conv2_output,
        nodes=nodes,
        initializers=initializers,
        conv_params=conv2_params,
        layer_name=f"{layer_name}_conv2"
    )

    # 残差连接
    if downsample or in_channels != out_channels:
        downsample_conv_weight_name = f"{layer_name}_downsample_conv_weight"
        downsample_conv_bias_name = f"{layer_name}_downsample_conv_bias"
        downsample_bn_scale_name = f"{layer_name}_downsample_bn_scale"
        downsample_bn_bias_name = f"{layer_name}_downsample_bn_bias"
        downsample_bn_mean_name = f"{layer_name}_downsample_bn_mean"
        downsample_bn_var_name = f"{layer_name}_downsample_bn_var"

        downsample_params = {
            'weight_shape': [out_channels, in_channels, 1, 1],
            'kernel_shape': [1, 1],
            'pads': [0, 0, 0, 0],
            'strides': [stride, stride]
        }

        downsample_output = f"{layer_name}_downsample_output"
        downsample_output = conv_bn_relu(
            input_name=input_name,
            conv_weight_name=downsample_conv_weight_name,
            conv_bias_name=downsample_conv_bias_name,
            bn_scale_name=downsample_bn_scale_name,
            bn_bias_name=downsample_bn_bias_name,
            bn_mean_name=downsample_bn_mean_name,
            bn_var_name=downsample_bn_var_name,
            output_name=downsample_output,
            nodes=nodes,
            initializers=initializers,
            conv_params=downsample_params,
            layer_name=f"{layer_name}_downsample"
        )
    else:
        downsample_output = input_name

    # Add 节点
    add_output = f"{layer_name}_add_output"
    add_node = helper.make_node(
        name=f"{layer_name}_add",
        op_type="Add",
        inputs=[conv2_output, downsample_output],
        outputs=[add_output]
    )
    nodes.append(add_node)

    # 最后的 ReLU
    relu_output = output_name
    relu_node = helper.make_node(
        name=f"{layer_name}_relu",
        op_type="Relu",
        inputs=[add_output],
        outputs=[relu_output]
    )
    nodes.append(relu_node)

    return relu_output

def main():
    """创建并保存ResNet-18 ONNX模型。"""
    # 定义输入和输出的形状
    batch_size = 1
    input_channels = 3
    input_height = 224
    input_width = 224
    num_classes = 1000

    input_shape = [batch_size, input_channels, input_height, input_width]
    output_shape = [batch_size, num_classes]

    # 创建输入和输出
    model_input_name = "input0"
    model_output_name = "output0"

    input = helper.make_tensor_value_info(model_input_name, TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(model_output_name, TensorProto.FLOAT, output_shape)

    nodes = []
    initializers = []
    previous_output_name = model_input_name

    # 初始卷积层
    conv1_weight_name = "conv1_weight"
    conv1_bias_name = "conv1_bias"
    bn1_scale_name = "bn1_scale"
    bn1_bias_name = "bn1_bias"
    bn1_mean_name = "bn1_mean"
    bn1_var_name = "bn1_var"

    conv1_params = {
        'weight_shape': [64, 3, 7, 7],
        'kernel_shape': [7, 7],
        'pads': [3, 3, 3, 3],
        'strides': [2, 2]
    }

    conv1_output = conv_bn_relu(
        input_name=previous_output_name,
        conv_weight_name=conv1_weight_name,
        conv_bias_name=conv1_bias_name,
        bn_scale_name=bn1_scale_name,
        bn_bias_name=bn1_bias_name,
        bn_mean_name=bn1_mean_name,
        bn_var_name=bn1_var_name,
        output_name="conv1_relu_output",
        nodes=nodes,
        initializers=initializers,
        conv_params=conv1_params,
        layer_name="conv1"
    )
    previous_output_name = conv1_output

    # MaxPool 层
    maxpool_output = "maxpool_output"
    maxpool_node = helper.make_node(
        name="maxpool",
        op_type="MaxPool",
        inputs=[previous_output_name],
        outputs=[maxpool_output],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1]
    )
    nodes.append(maxpool_node)
    previous_output_name = maxpool_output

    # 定义每个阶段的层数和通道数
    layers = [2, 2, 2, 2]  # ResNet-18
    channels = [64, 128, 256, 512]

    in_channels = 64
    for stage, num_blocks in enumerate(layers):
        out_channels = channels[stage]
        for block in range(num_blocks):
            stride = 2 if block == 0 and stage != 0 else 1
            downsample = stride == 2
            layer_name = f"layer{stage + 1}_{block + 1}"
            previous_output_name = basic_block(
                input_name=previous_output_name,
                output_name=f"{layer_name}_output",
                nodes=nodes,
                initializers=initializers,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                layer_name=layer_name,
                downsample=downsample
            )
            in_channels = out_channels

    # 全局平均池化层
    avgpool_output = "avgpool_output"
    avgpool_node = helper.make_node(
        name="avgpool",
        op_type="GlobalAveragePool",
        inputs=[previous_output_name],
        outputs=[avgpool_output]
    )
    nodes.append(avgpool_node)

    # Flatten 层
    flatten_output = "flatten_output"
    flatten_node = helper.make_node(
        name="flatten",
        op_type="Flatten",
        inputs=[avgpool_output],
        outputs=[flatten_output],
        axis=1
    )
    nodes.append(flatten_node)

    # 全连接层
    fc_weight_name = "fc_weight"
    fc_bias_name = "fc_bias"
    fc_weight = np.random.randn(num_classes, in_channels).astype(np.float32) * np.sqrt(2.0 / (in_channels + num_classes))
    fc_bias = np.zeros((num_classes,), dtype=np.float32)

    fc_weight_initializer = create_initializer_tensor(name=fc_weight_name, tensor_array=fc_weight)
    fc_bias_initializer = create_initializer_tensor(name=fc_bias_name, tensor_array=fc_bias)
    initializers.extend([fc_weight_initializer, fc_bias_initializer])

    fc_output = "fc_output"
    fc_node = helper.make_node(
        name="fc",
        op_type="Gemm",
        inputs=[flatten_output, fc_weight_name, fc_bias_name],
        outputs=[fc_output],
        alpha=1.0,
        beta=1.0,
        transB=1
    )
    nodes.append(fc_node)

    # 输出层
    output_node = helper.make_node(
        name="output",
        op_type="Identity",
        inputs=[fc_output],
        outputs=[model_output_name]
    )
    nodes.append(output_node)

    # 创建计算图
    graph = helper.make_graph(
        name="resnet18",
        inputs=[input],
        outputs=[output],
        nodes=nodes,
        initializer=initializers
    )

    # 创建模型
    model = helper.make_model(graph, producer_name="onnx-resnet-sample")
    model.opset_import[0].version = 12

    # 验证并保存模型
    try:
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, full_check=True)
        output_path = get_onnx_path(__file__, "resnet18_zh.onnx")
        onnx.save(model, output_path)
        print(f"成功创建 {output_path}")
        parse_onnx(model)
    except Exception as e:
        print(f"创建或保存模型失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# python3 3.read-and-parse-onnx/7.create_onnx_resnet_zh.py
# conv1 conv: input=input0, weight=[64, 3, 7, 7], output=conv1_conv_output
# layer1_1_conv1 conv: input=maxpool_output, weight=[64, 64, 3, 3], output=layer1_1_conv1_conv_output
# layer1_1_conv2 conv: input=layer1_1_relu1_output, weight=[64, 64, 3, 3], output=layer1_1_conv2_conv_output
# layer1_2_conv1 conv: input=layer1_1_output, weight=[64, 64, 3, 3], output=layer1_2_conv1_conv_output
# layer1_2_conv2 conv: input=layer1_2_relu1_output, weight=[64, 64, 3, 3], output=layer1_2_conv2_conv_output
# layer2_1_conv1 conv: input=layer1_2_output, weight=[128, 64, 3, 3], output=layer2_1_conv1_conv_output
# layer2_1_conv2 conv: input=layer2_1_relu1_output, weight=[128, 128, 3, 3], output=layer2_1_conv2_conv_output
# layer2_1_downsample conv: input=layer1_2_output, weight=[128, 64, 1, 1], output=layer2_1_downsample_conv_output
# layer2_2_conv1 conv: input=layer2_1_output, weight=[128, 128, 3, 3], output=layer2_2_conv1_conv_output
# layer2_2_conv2 conv: input=layer2_2_relu1_output, weight=[128, 128, 3, 3], output=layer2_2_conv2_conv_output
# layer3_1_conv1 conv: input=layer2_2_output, weight=[256, 128, 3, 3], output=layer3_1_conv1_conv_output
# layer3_1_conv2 conv: input=layer3_1_relu1_output, weight=[256, 256, 3, 3], output=layer3_1_conv2_conv_output
# layer3_1_downsample conv: input=layer2_2_output, weight=[256, 128, 1, 1], output=layer3_1_downsample_conv_output
# layer3_2_conv1 conv: input=layer3_1_output, weight=[256, 256, 3, 3], output=layer3_2_conv1_conv_output
# layer3_2_conv2 conv: input=layer3_2_relu1_output, weight=[256, 256, 3, 3], output=layer3_2_conv2_conv_output
# layer4_1_conv1 conv: input=layer3_2_output, weight=[512, 256, 3, 3], output=layer4_1_conv1_conv_output
# layer4_1_conv2 conv: input=layer4_1_relu1_output, weight=[512, 512, 3, 3], output=layer4_1_conv2_conv_output
# layer4_1_downsample conv: input=layer3_2_output, weight=[512, 256, 1, 1], output=layer4_1_downsample_conv_output
# layer4_2_conv1 conv: input=layer4_1_output, weight=[512, 512, 3, 3], output=layer4_2_conv1_conv_output
# layer4_2_conv2 conv: input=layer4_2_relu1_output, weight=[512, 512, 3, 3], output=layer4_2_conv2_conv_output
# 成功创建 /home/wudi/work/github/WuChenDi/TensorRT-ONNX/3.read-and-parse-onnx/models/resnet18_zh.onnx

# ==================================================
# 解析输入信息
# ==================================================
# 输入信息:
#   名称:     input0
#   数据类型: FLOAT
#   形状:     [1, 3, 224, 224]

# ==================================================
# 解析输出信息
# ==================================================
# 输出信息:
#   名称:     output0
#   数据类型: FLOAT
#   形状:     [1, 1000]

# ==================================================
# 解析节点信息
# ==================================================
# 节点信息:
#   名称:     conv1_conv
#   操作类型: Conv
#   输入:     ['input0', 'conv1_weight', 'conv1_bias']
#   输出:     ['conv1_conv_output']
#   属性:     ['kernel_shape: [7, 7]', 'pads: [3, 3, 3, 3]', 'strides: [2, 2]']
# 节点信息:
#   名称:     conv1_bn
#   操作类型: BatchNormalization
#   输入:     ['conv1_conv_output', 'bn1_scale', 'bn1_bias', 'bn1_mean', 'bn1_var']
#   输出:     ['conv1_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     conv1_relu
#   操作类型: Relu
#   输入:     ['conv1_bn_output']
#   输出:     ['conv1_relu_output']
#   属性:     []
# 节点信息:
#   名称:     maxpool
#   操作类型: MaxPool
#   输入:     ['conv1_relu_output']
#   输出:     ['maxpool_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [2, 2]']
# 节点信息:
#   名称:     layer1_1_conv1_conv
#   操作类型: Conv
#   输入:     ['maxpool_output', 'layer1_1_conv1_weight', 'layer1_1_conv1_bias']
#   输出:     ['layer1_1_conv1_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer1_1_conv1_bn
#   操作类型: BatchNormalization
#   输入:     ['layer1_1_conv1_conv_output', 'layer1_1_bn1_scale', 'layer1_1_bn1_bias', 'layer1_1_bn1_mean', 'layer1_1_bn1_var']
#   输出:     ['layer1_1_conv1_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer1_1_conv1_relu
#   操作类型: Relu
#   输入:     ['layer1_1_conv1_bn_output']
#   输出:     ['layer1_1_relu1_output']
#   属性:     []
# 节点信息:
#   名称:     layer1_1_conv2_conv
#   操作类型: Conv
#   输入:     ['layer1_1_relu1_output', 'layer1_1_conv2_weight', 'layer1_1_conv2_bias']
#   输出:     ['layer1_1_conv2_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer1_1_conv2_bn
#   操作类型: BatchNormalization
#   输入:     ['layer1_1_conv2_conv_output', 'layer1_1_bn2_scale', 'layer1_1_bn2_bias', 'layer1_1_bn2_mean', 'layer1_1_bn2_var']
#   输出:     ['layer1_1_conv2_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer1_1_conv2_relu
#   操作类型: Relu
#   输入:     ['layer1_1_conv2_bn_output']
#   输出:     ['layer1_1_conv2_output']
#   属性:     []
# 节点信息:
#   名称:     layer1_1_add
#   操作类型: Add
#   输入:     ['layer1_1_conv2_output', 'maxpool_output']
#   输出:     ['layer1_1_add_output']
#   属性:     []
# 节点信息:
#   名称:     layer1_1_relu
#   操作类型: Relu
#   输入:     ['layer1_1_add_output']
#   输出:     ['layer1_1_output']
#   属性:     []
# 节点信息:
#   名称:     layer1_2_conv1_conv
#   操作类型: Conv
#   输入:     ['layer1_1_output', 'layer1_2_conv1_weight', 'layer1_2_conv1_bias']
#   输出:     ['layer1_2_conv1_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer1_2_conv1_bn
#   操作类型: BatchNormalization
#   输入:     ['layer1_2_conv1_conv_output', 'layer1_2_bn1_scale', 'layer1_2_bn1_bias', 'layer1_2_bn1_mean', 'layer1_2_bn1_var']
#   输出:     ['layer1_2_conv1_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer1_2_conv1_relu
#   操作类型: Relu
#   输入:     ['layer1_2_conv1_bn_output']
#   输出:     ['layer1_2_relu1_output']
#   属性:     []
# 节点信息:
#   名称:     layer1_2_conv2_conv
#   操作类型: Conv
#   输入:     ['layer1_2_relu1_output', 'layer1_2_conv2_weight', 'layer1_2_conv2_bias']
#   输出:     ['layer1_2_conv2_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer1_2_conv2_bn
#   操作类型: BatchNormalization
#   输入:     ['layer1_2_conv2_conv_output', 'layer1_2_bn2_scale', 'layer1_2_bn2_bias', 'layer1_2_bn2_mean', 'layer1_2_bn2_var']
#   输出:     ['layer1_2_conv2_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer1_2_conv2_relu
#   操作类型: Relu
#   输入:     ['layer1_2_conv2_bn_output']
#   输出:     ['layer1_2_conv2_output']
#   属性:     []
# 节点信息:
#   名称:     layer1_2_add
#   操作类型: Add
#   输入:     ['layer1_2_conv2_output', 'layer1_1_output']
#   输出:     ['layer1_2_add_output']
#   属性:     []
# 节点信息:
#   名称:     layer1_2_relu
#   操作类型: Relu
#   输入:     ['layer1_2_add_output']
#   输出:     ['layer1_2_output']
#   属性:     []
# 节点信息:
#   名称:     layer2_1_conv1_conv
#   操作类型: Conv
#   输入:     ['layer1_2_output', 'layer2_1_conv1_weight', 'layer2_1_conv1_bias']
#   输出:     ['layer2_1_conv1_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [2, 2]']
# 节点信息:
#   名称:     layer2_1_conv1_bn
#   操作类型: BatchNormalization
#   输入:     ['layer2_1_conv1_conv_output', 'layer2_1_bn1_scale', 'layer2_1_bn1_bias', 'layer2_1_bn1_mean', 'layer2_1_bn1_var']
#   输出:     ['layer2_1_conv1_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer2_1_conv1_relu
#   操作类型: Relu
#   输入:     ['layer2_1_conv1_bn_output']
#   输出:     ['layer2_1_relu1_output']
#   属性:     []
# 节点信息:
#   名称:     layer2_1_conv2_conv
#   操作类型: Conv
#   输入:     ['layer2_1_relu1_output', 'layer2_1_conv2_weight', 'layer2_1_conv2_bias']
#   输出:     ['layer2_1_conv2_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer2_1_conv2_bn
#   操作类型: BatchNormalization
#   输入:     ['layer2_1_conv2_conv_output', 'layer2_1_bn2_scale', 'layer2_1_bn2_bias', 'layer2_1_bn2_mean', 'layer2_1_bn2_var']
#   输出:     ['layer2_1_conv2_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer2_1_conv2_relu
#   操作类型: Relu
#   输入:     ['layer2_1_conv2_bn_output']
#   输出:     ['layer2_1_conv2_output']
#   属性:     []
# 节点信息:
#   名称:     layer2_1_downsample_conv
#   操作类型: Conv
#   输入:     ['layer1_2_output', 'layer2_1_downsample_conv_weight', 'layer2_1_downsample_conv_bias']
#   输出:     ['layer2_1_downsample_conv_output']
#   属性:     ['kernel_shape: [1, 1]', 'pads: [0, 0, 0, 0]', 'strides: [2, 2]']
# 节点信息:
#   名称:     layer2_1_downsample_bn
#   操作类型: BatchNormalization
#   输入:     ['layer2_1_downsample_conv_output', 'layer2_1_downsample_bn_scale', 'layer2_1_downsample_bn_bias', 'layer2_1_downsample_bn_mean', 'layer2_1_downsample_bn_var']
#   输出:     ['layer2_1_downsample_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer2_1_downsample_relu
#   操作类型: Relu
#   输入:     ['layer2_1_downsample_bn_output']
#   输出:     ['layer2_1_downsample_output']
#   属性:     []
# 节点信息:
#   名称:     layer2_1_add
#   操作类型: Add
#   输入:     ['layer2_1_conv2_output', 'layer2_1_downsample_output']
#   输出:     ['layer2_1_add_output']
#   属性:     []
# 节点信息:
#   名称:     layer2_1_relu
#   操作类型: Relu
#   输入:     ['layer2_1_add_output']
#   输出:     ['layer2_1_output']
#   属性:     []
# 节点信息:
#   名称:     layer2_2_conv1_conv
#   操作类型: Conv
#   输入:     ['layer2_1_output', 'layer2_2_conv1_weight', 'layer2_2_conv1_bias']
#   输出:     ['layer2_2_conv1_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer2_2_conv1_bn
#   操作类型: BatchNormalization
#   输入:     ['layer2_2_conv1_conv_output', 'layer2_2_bn1_scale', 'layer2_2_bn1_bias', 'layer2_2_bn1_mean', 'layer2_2_bn1_var']
#   输出:     ['layer2_2_conv1_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer2_2_conv1_relu
#   操作类型: Relu
#   输入:     ['layer2_2_conv1_bn_output']
#   输出:     ['layer2_2_relu1_output']
#   属性:     []
# 节点信息:
#   名称:     layer2_2_conv2_conv
#   操作类型: Conv
#   输入:     ['layer2_2_relu1_output', 'layer2_2_conv2_weight', 'layer2_2_conv2_bias']
#   输出:     ['layer2_2_conv2_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer2_2_conv2_bn
#   操作类型: BatchNormalization
#   输入:     ['layer2_2_conv2_conv_output', 'layer2_2_bn2_scale', 'layer2_2_bn2_bias', 'layer2_2_bn2_mean', 'layer2_2_bn2_var']
#   输出:     ['layer2_2_conv2_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer2_2_conv2_relu
#   操作类型: Relu
#   输入:     ['layer2_2_conv2_bn_output']
#   输出:     ['layer2_2_conv2_output']
#   属性:     []
# 节点信息:
#   名称:     layer2_2_add
#   操作类型: Add
#   输入:     ['layer2_2_conv2_output', 'layer2_1_output']
#   输出:     ['layer2_2_add_output']
#   属性:     []
# 节点信息:
#   名称:     layer2_2_relu
#   操作类型: Relu
#   输入:     ['layer2_2_add_output']
#   输出:     ['layer2_2_output']
#   属性:     []
# 节点信息:
#   名称:     layer3_1_conv1_conv
#   操作类型: Conv
#   输入:     ['layer2_2_output', 'layer3_1_conv1_weight', 'layer3_1_conv1_bias']
#   输出:     ['layer3_1_conv1_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [2, 2]']
# 节点信息:
#   名称:     layer3_1_conv1_bn
#   操作类型: BatchNormalization
#   输入:     ['layer3_1_conv1_conv_output', 'layer3_1_bn1_scale', 'layer3_1_bn1_bias', 'layer3_1_bn1_mean', 'layer3_1_bn1_var']
#   输出:     ['layer3_1_conv1_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer3_1_conv1_relu
#   操作类型: Relu
#   输入:     ['layer3_1_conv1_bn_output']
#   输出:     ['layer3_1_relu1_output']
#   属性:     []
# 节点信息:
#   名称:     layer3_1_conv2_conv
#   操作类型: Conv
#   输入:     ['layer3_1_relu1_output', 'layer3_1_conv2_weight', 'layer3_1_conv2_bias']
#   输出:     ['layer3_1_conv2_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer3_1_conv2_bn
#   操作类型: BatchNormalization
#   输入:     ['layer3_1_conv2_conv_output', 'layer3_1_bn2_scale', 'layer3_1_bn2_bias', 'layer3_1_bn2_mean', 'layer3_1_bn2_var']
#   输出:     ['layer3_1_conv2_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer3_1_conv2_relu
#   操作类型: Relu
#   输入:     ['layer3_1_conv2_bn_output']
#   输出:     ['layer3_1_conv2_output']
#   属性:     []
# 节点信息:
#   名称:     layer3_1_downsample_conv
#   操作类型: Conv
#   输入:     ['layer2_2_output', 'layer3_1_downsample_conv_weight', 'layer3_1_downsample_conv_bias']
#   输出:     ['layer3_1_downsample_conv_output']
#   属性:     ['kernel_shape: [1, 1]', 'pads: [0, 0, 0, 0]', 'strides: [2, 2]']
# 节点信息:
#   名称:     layer3_1_downsample_bn
#   操作类型: BatchNormalization
#   输入:     ['layer3_1_downsample_conv_output', 'layer3_1_downsample_bn_scale', 'layer3_1_downsample_bn_bias', 'layer3_1_downsample_bn_mean', 'layer3_1_downsample_bn_var']
#   输出:     ['layer3_1_downsample_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer3_1_downsample_relu
#   操作类型: Relu
#   输入:     ['layer3_1_downsample_bn_output']
#   输出:     ['layer3_1_downsample_output']
#   属性:     []
# 节点信息:
#   名称:     layer3_1_add
#   操作类型: Add
#   输入:     ['layer3_1_conv2_output', 'layer3_1_downsample_output']
#   输出:     ['layer3_1_add_output']
#   属性:     []
# 节点信息:
#   名称:     layer3_1_relu
#   操作类型: Relu
#   输入:     ['layer3_1_add_output']
#   输出:     ['layer3_1_output']
#   属性:     []
# 节点信息:
#   名称:     layer3_2_conv1_conv
#   操作类型: Conv
#   输入:     ['layer3_1_output', 'layer3_2_conv1_weight', 'layer3_2_conv1_bias']
#   输出:     ['layer3_2_conv1_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer3_2_conv1_bn
#   操作类型: BatchNormalization
#   输入:     ['layer3_2_conv1_conv_output', 'layer3_2_bn1_scale', 'layer3_2_bn1_bias', 'layer3_2_bn1_mean', 'layer3_2_bn1_var']
#   输出:     ['layer3_2_conv1_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer3_2_conv1_relu
#   操作类型: Relu
#   输入:     ['layer3_2_conv1_bn_output']
#   输出:     ['layer3_2_relu1_output']
#   属性:     []
# 节点信息:
#   名称:     layer3_2_conv2_conv
#   操作类型: Conv
#   输入:     ['layer3_2_relu1_output', 'layer3_2_conv2_weight', 'layer3_2_conv2_bias']
#   输出:     ['layer3_2_conv2_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer3_2_conv2_bn
#   操作类型: BatchNormalization
#   输入:     ['layer3_2_conv2_conv_output', 'layer3_2_bn2_scale', 'layer3_2_bn2_bias', 'layer3_2_bn2_mean', 'layer3_2_bn2_var']
#   输出:     ['layer3_2_conv2_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer3_2_conv2_relu
#   操作类型: Relu
#   输入:     ['layer3_2_conv2_bn_output']
#   输出:     ['layer3_2_conv2_output']
#   属性:     []
# 节点信息:
#   名称:     layer3_2_add
#   操作类型: Add
#   输入:     ['layer3_2_conv2_output', 'layer3_1_output']
#   输出:     ['layer3_2_add_output']
#   属性:     []
# 节点信息:
#   名称:     layer3_2_relu
#   操作类型: Relu
#   输入:     ['layer3_2_add_output']
#   输出:     ['layer3_2_output']
#   属性:     []
# 节点信息:
#   名称:     layer4_1_conv1_conv
#   操作类型: Conv
#   输入:     ['layer3_2_output', 'layer4_1_conv1_weight', 'layer4_1_conv1_bias']
#   输出:     ['layer4_1_conv1_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [2, 2]']
# 节点信息:
#   名称:     layer4_1_conv1_bn
#   操作类型: BatchNormalization
#   输入:     ['layer4_1_conv1_conv_output', 'layer4_1_bn1_scale', 'layer4_1_bn1_bias', 'layer4_1_bn1_mean', 'layer4_1_bn1_var']
#   输出:     ['layer4_1_conv1_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer4_1_conv1_relu
#   操作类型: Relu
#   输入:     ['layer4_1_conv1_bn_output']
#   输出:     ['layer4_1_relu1_output']
#   属性:     []
# 节点信息:
#   名称:     layer4_1_conv2_conv
#   操作类型: Conv
#   输入:     ['layer4_1_relu1_output', 'layer4_1_conv2_weight', 'layer4_1_conv2_bias']
#   输出:     ['layer4_1_conv2_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer4_1_conv2_bn
#   操作类型: BatchNormalization
#   输入:     ['layer4_1_conv2_conv_output', 'layer4_1_bn2_scale', 'layer4_1_bn2_bias', 'layer4_1_bn2_mean', 'layer4_1_bn2_var']
#   输出:     ['layer4_1_conv2_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer4_1_conv2_relu
#   操作类型: Relu
#   输入:     ['layer4_1_conv2_bn_output']
#   输出:     ['layer4_1_conv2_output']
#   属性:     []
# 节点信息:
#   名称:     layer4_1_downsample_conv
#   操作类型: Conv
#   输入:     ['layer3_2_output', 'layer4_1_downsample_conv_weight', 'layer4_1_downsample_conv_bias']
#   输出:     ['layer4_1_downsample_conv_output']
#   属性:     ['kernel_shape: [1, 1]', 'pads: [0, 0, 0, 0]', 'strides: [2, 2]']
# 节点信息:
#   名称:     layer4_1_downsample_bn
#   操作类型: BatchNormalization
#   输入:     ['layer4_1_downsample_conv_output', 'layer4_1_downsample_bn_scale', 'layer4_1_downsample_bn_bias', 'layer4_1_downsample_bn_mean', 'layer4_1_downsample_bn_var']
#   输出:     ['layer4_1_downsample_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer4_1_downsample_relu
#   操作类型: Relu
#   输入:     ['layer4_1_downsample_bn_output']
#   输出:     ['layer4_1_downsample_output']
#   属性:     []
# 节点信息:
#   名称:     layer4_1_add
#   操作类型: Add
#   输入:     ['layer4_1_conv2_output', 'layer4_1_downsample_output']
#   输出:     ['layer4_1_add_output']
#   属性:     []
# 节点信息:
#   名称:     layer4_1_relu
#   操作类型: Relu
#   输入:     ['layer4_1_add_output']
#   输出:     ['layer4_1_output']
#   属性:     []
# 节点信息:
#   名称:     layer4_2_conv1_conv
#   操作类型: Conv
#   输入:     ['layer4_1_output', 'layer4_2_conv1_weight', 'layer4_2_conv1_bias']
#   输出:     ['layer4_2_conv1_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer4_2_conv1_bn
#   操作类型: BatchNormalization
#   输入:     ['layer4_2_conv1_conv_output', 'layer4_2_bn1_scale', 'layer4_2_bn1_bias', 'layer4_2_bn1_mean', 'layer4_2_bn1_var']
#   输出:     ['layer4_2_conv1_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer4_2_conv1_relu
#   操作类型: Relu
#   输入:     ['layer4_2_conv1_bn_output']
#   输出:     ['layer4_2_relu1_output']
#   属性:     []
# 节点信息:
#   名称:     layer4_2_conv2_conv
#   操作类型: Conv
#   输入:     ['layer4_2_relu1_output', 'layer4_2_conv2_weight', 'layer4_2_conv2_bias']
#   输出:     ['layer4_2_conv2_conv_output']
#   属性:     ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# 节点信息:
#   名称:     layer4_2_conv2_bn
#   操作类型: BatchNormalization
#   输入:     ['layer4_2_conv2_conv_output', 'layer4_2_bn2_scale', 'layer4_2_bn2_bias', 'layer4_2_bn2_mean', 'layer4_2_bn2_var']
#   输出:     ['layer4_2_conv2_bn_output']
#   属性:     ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# 节点信息:
#   名称:     layer4_2_conv2_relu
#   操作类型: Relu
#   输入:     ['layer4_2_conv2_bn_output']
#   输出:     ['layer4_2_conv2_output']
#   属性:     []
# 节点信息:
#   名称:     layer4_2_add
#   操作类型: Add
#   输入:     ['layer4_2_conv2_output', 'layer4_1_output']
#   输出:     ['layer4_2_add_output']
#   属性:     []
# 节点信息:
#   名称:     layer4_2_relu
#   操作类型: Relu
#   输入:     ['layer4_2_add_output']
#   输出:     ['layer4_2_output']
#   属性:     []
# 节点信息:
#   名称:     avgpool
#   操作类型: GlobalAveragePool
#   输入:     ['layer4_2_output']
#   输出:     ['avgpool_output']
#   属性:     []
# 节点信息:
#   名称:     flatten
#   操作类型: Flatten
#   输入:     ['avgpool_output']
#   输出:     ['flatten_output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     fc
#   操作类型: Gemm
#   输入:     ['flatten_output', 'fc_weight', 'fc_bias']
#   输出:     ['fc_output']
#   属性:     ['alpha: 1.0', 'beta: 1.0', 'transB: 不支持']
# 节点信息:
#   名称:     output
#   操作类型: Identity
#   输入:     ['fc_output']
#   输出:     ['output0']
#   属性:     []

# ==================================================
# 解析初始化权重信息
# ==================================================
# 初始化权重信息:
#   名称:     conv1_weight
#   数据类型: FLOAT
#   形状:     [64, 3, 7, 7]
# 初始化权重信息:
#   名称:     conv1_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     bn1_scale
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     bn1_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     bn1_mean
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     bn1_var
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_conv1_weight
#   数据类型: FLOAT
#   形状:     [64, 64, 3, 3]
# 初始化权重信息:
#   名称:     layer1_1_conv1_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_bn1_scale
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_bn1_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_bn1_mean
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_bn1_var
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_conv2_weight
#   数据类型: FLOAT
#   形状:     [64, 64, 3, 3]
# 初始化权重信息:
#   名称:     layer1_1_conv2_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_bn2_scale
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_bn2_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_bn2_mean
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_1_bn2_var
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_conv1_weight
#   数据类型: FLOAT
#   形状:     [64, 64, 3, 3]
# 初始化权重信息:
#   名称:     layer1_2_conv1_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_bn1_scale
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_bn1_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_bn1_mean
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_bn1_var
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_conv2_weight
#   数据类型: FLOAT
#   形状:     [64, 64, 3, 3]
# 初始化权重信息:
#   名称:     layer1_2_conv2_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_bn2_scale
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_bn2_bias
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_bn2_mean
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer1_2_bn2_var
#   数据类型: FLOAT
#   形状:     [64]
# 初始化权重信息:
#   名称:     layer2_1_conv1_weight
#   数据类型: FLOAT
#   形状:     [128, 64, 3, 3]
# 初始化权重信息:
#   名称:     layer2_1_conv1_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_bn1_scale
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_bn1_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_bn1_mean
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_bn1_var
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_conv2_weight
#   数据类型: FLOAT
#   形状:     [128, 128, 3, 3]
# 初始化权重信息:
#   名称:     layer2_1_conv2_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_bn2_scale
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_bn2_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_bn2_mean
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_bn2_var
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_downsample_conv_weight
#   数据类型: FLOAT
#   形状:     [128, 64, 1, 1]
# 初始化权重信息:
#   名称:     layer2_1_downsample_conv_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_downsample_bn_scale
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_downsample_bn_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_downsample_bn_mean
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_1_downsample_bn_var
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_conv1_weight
#   数据类型: FLOAT
#   形状:     [128, 128, 3, 3]
# 初始化权重信息:
#   名称:     layer2_2_conv1_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_bn1_scale
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_bn1_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_bn1_mean
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_bn1_var
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_conv2_weight
#   数据类型: FLOAT
#   形状:     [128, 128, 3, 3]
# 初始化权重信息:
#   名称:     layer2_2_conv2_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_bn2_scale
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_bn2_bias
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_bn2_mean
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer2_2_bn2_var
#   数据类型: FLOAT
#   形状:     [128]
# 初始化权重信息:
#   名称:     layer3_1_conv1_weight
#   数据类型: FLOAT
#   形状:     [256, 128, 3, 3]
# 初始化权重信息:
#   名称:     layer3_1_conv1_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_bn1_scale
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_bn1_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_bn1_mean
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_bn1_var
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_conv2_weight
#   数据类型: FLOAT
#   形状:     [256, 256, 3, 3]
# 初始化权重信息:
#   名称:     layer3_1_conv2_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_bn2_scale
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_bn2_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_bn2_mean
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_bn2_var
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_downsample_conv_weight
#   数据类型: FLOAT
#   形状:     [256, 128, 1, 1]
# 初始化权重信息:
#   名称:     layer3_1_downsample_conv_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_downsample_bn_scale
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_downsample_bn_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_downsample_bn_mean
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_1_downsample_bn_var
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_conv1_weight
#   数据类型: FLOAT
#   形状:     [256, 256, 3, 3]
# 初始化权重信息:
#   名称:     layer3_2_conv1_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_bn1_scale
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_bn1_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_bn1_mean
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_bn1_var
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_conv2_weight
#   数据类型: FLOAT
#   形状:     [256, 256, 3, 3]
# 初始化权重信息:
#   名称:     layer3_2_conv2_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_bn2_scale
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_bn2_bias
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_bn2_mean
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer3_2_bn2_var
#   数据类型: FLOAT
#   形状:     [256]
# 初始化权重信息:
#   名称:     layer4_1_conv1_weight
#   数据类型: FLOAT
#   形状:     [512, 256, 3, 3]
# 初始化权重信息:
#   名称:     layer4_1_conv1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_bn1_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_bn1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_bn1_mean
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_bn1_var
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_conv2_weight
#   数据类型: FLOAT
#   形状:     [512, 512, 3, 3]
# 初始化权重信息:
#   名称:     layer4_1_conv2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_bn2_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_bn2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_bn2_mean
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_bn2_var
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_downsample_conv_weight
#   数据类型: FLOAT
#   形状:     [512, 256, 1, 1]
# 初始化权重信息:
#   名称:     layer4_1_downsample_conv_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_downsample_bn_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_downsample_bn_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_downsample_bn_mean
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_1_downsample_bn_var
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_conv1_weight
#   数据类型: FLOAT
#   形状:     [512, 512, 3, 3]
# 初始化权重信息:
#   名称:     layer4_2_conv1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_bn1_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_bn1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_bn1_mean
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_bn1_var
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_conv2_weight
#   数据类型: FLOAT
#   形状:     [512, 512, 3, 3]
# 初始化权重信息:
#   名称:     layer4_2_conv2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_bn2_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_bn2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_bn2_mean
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_2_bn2_var
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     fc_weight
#   数据类型: FLOAT
#   形状:     [1000, 512]
# 初始化权重信息:
#   名称:     fc_bias
#   数据类型: FLOAT
#   形状:     [1000]
