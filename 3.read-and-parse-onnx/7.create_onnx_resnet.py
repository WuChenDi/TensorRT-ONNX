# Create a ResNet-18 ONNX model
# - Includes initial convolution, max pooling, 4 residual stages, global average pooling, and fully connected layer
# - Uses opset_version=12 for compatibility
# - Saves model to models/resnet18.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from utils import get_onnx_path

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: int = TensorProto.FLOAT
) -> onnx.TensorProto:
    """Create an ONNX TensorProto initializer."""
    if tensor_array.size == 0:
        raise ValueError(f"Initializer {name} is empty")
    initializer = helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())
    return initializer

def parse_onnx(model: onnx.ModelProto) -> None:
    """Parse and print ONNX model structure."""
    try:
        graph = model.graph
        if not graph:
            raise ValueError("Model graph is empty")

        print(f"\n{'='*50}\nParsing Input Information\n{'='*50}")
        for input in graph.input:
            input_shape = [d.dim_value if d.dim_value != 0 else None for d in input.type.tensor_type.shape.dim]
            print(f"Input Info:\n  Name:     {input.name}\n  Data Type: {TensorProto.DataType.Name(input.type.tensor_type.elem_type)}\n  Shape:     {input_shape}")

        print(f"\n{'='*50}\nParsing Output Information\n{'='*50}")
        for output in graph.output:
            output_shape = [d.dim_value if d.dim_value != 0 else None for d in output.type.tensor_type.shape.dim]
            print(f"Output Info:\n  Name:     {output.name}\n  Data Type: {TensorProto.DataType.Name(output.type.tensor_type.elem_type)}\n  Shape:     {output_shape}")

        print(f"\n{'='*50}\nParsing Node Information\n{'='*50}")
        for node in graph.node:
            attributes = [f"{attr.name}: {attr.ints if attr.type == onnx.AttributeProto.INTS else attr.floats if attr.type == onnx.AttributeProto.FLOATS else attr.f if attr.type == onnx.AttributeProto.FLOAT else attr.s.decode() if attr.type == onnx.AttributeProto.STRING else 'unsupported'}" for attr in node.attribute]
            print(f"Node Info:\n  Name:     {node.name}\n  Op Type:  {node.op_type}\n  Inputs:   {node.input}\n  Outputs:  {node.output}\n  Attributes: {attributes}")

        print(f"\n{'='*50}\nParsing Initializer Information\n{'='*50}")
        for initializer in graph.initializer:
            print(f"Initializer Info:\n  Name:     {initializer.name}\n  Data Type: {TensorProto.DataType.Name(initializer.data_type)}\n  Shape:     {list(initializer.dims)}")
    except Exception as e:
        print(f"Failed to parse model: {str(e)}")

def conv_bn_relu(input_name, conv_weight_name, conv_bias_name, bn_scale_name, bn_bias_name,
                 bn_mean_name, bn_var_name, output_name, nodes, initializers,
                 conv_params, layer_name):
    """Create a Conv -> BatchNorm -> ReLU combination."""
    conv_output = f"{layer_name}_conv_output"
    bn_output = f"{layer_name}_bn_output"
    relu_output = output_name

    # Convolution weights and bias (Xavier initialization)
    fan_in = conv_params['weight_shape'][1] * conv_params['kernel_shape'][0] * conv_params['kernel_shape'][1]
    fan_out = conv_params['weight_shape'][0]
    conv_weight = np.random.randn(*conv_params['weight_shape']).astype(np.float32) * np.sqrt(2.0 / (fan_in + fan_out))
    conv_bias = np.zeros((conv_params['weight_shape'][0],), dtype=np.float32)

    conv_weight_initializer = create_initializer_tensor(name=conv_weight_name, tensor_array=conv_weight)
    conv_bias_initializer = create_initializer_tensor(name=conv_bias_name, tensor_array=conv_bias)
    initializers.extend([conv_weight_initializer, conv_bias_initializer])

    # Convolution node
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

    # BatchNorm parameters
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

    # BatchNorm node
    bn_node = helper.make_node(
        name=f"{layer_name}_bn",
        op_type="BatchNormalization",
        inputs=[conv_output, bn_scale_name, bn_bias_name, bn_mean_name, bn_var_name],
        outputs=[bn_output],
        epsilon=1e-5,
        momentum=0.9
    )
    nodes.append(bn_node)

    # ReLU node
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
    """Create a ResNet BasicBlock (residual block)."""
    # First convolution layer
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

    # Second convolution layer
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

    # Residual connection
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

    # Add node
    add_output = f"{layer_name}_add_output"
    add_node = helper.make_node(
        name=f"{layer_name}_add",
        op_type="Add",
        inputs=[conv2_output, downsample_output],
        outputs=[add_output]
    )
    nodes.append(add_node)

    # Final ReLU
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
    """Create and save ResNet-18 ONNX model."""
    # Define input and output shapes
    batch_size = 1
    input_channels = 3
    input_height = 224
    input_width = 224
    num_classes = 1000

    input_shape = [batch_size, input_channels, input_height, input_width]
    output_shape = [batch_size, num_classes]

    # Create input and output
    model_input_name = "input0"
    model_output_name = "output0"

    input = helper.make_tensor_value_info(model_input_name, TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(model_output_name, TensorProto.FLOAT, output_shape)

    nodes = []
    initializers = []
    previous_output_name = model_input_name

    # Initial convolution layer
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

    # MaxPool layer
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

    # Define stages and channels
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

    # Global average pooling
    avgpool_output = "avgpool_output"
    avgpool_node = helper.make_node(
        name="avgpool",
        op_type="GlobalAveragePool",
        inputs=[previous_output_name],
        outputs=[avgpool_output]
    )
    nodes.append(avgpool_node)

    # Flatten layer
    flatten_output = "flatten_output"
    flatten_node = helper.make_node(
        name="flatten",
        op_type="Flatten",
        inputs=[avgpool_output],
        outputs=[flatten_output],
        axis=1
    )
    nodes.append(flatten_node)

    # Fully connected layer
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

    # Output layer
    output_node = helper.make_node(
        name="output",
        op_type="Identity",
        inputs=[fc_output],
        outputs=[model_output_name]
    )
    nodes.append(output_node)

    # Create graph
    graph = helper.make_graph(
        name="resnet18",
        inputs=[input],
        outputs=[output],
        nodes=nodes,
        initializer=initializers
    )

    # Create model
    model = helper.make_model(graph, producer_name="onnx-resnet-sample")
    model.opset_import[0].version = 12

    # Validate and save model
    try:
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, full_check=True)
        output_path = get_onnx_path(__file__, "resnet18.onnx")
        onnx.save(model, output_path)
        print(f"Successfully created {output_path}")
        parse_onnx(model)
    except Exception as e:
        print(f"Failed to create or save model: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# python3 3.read-and-parse-onnx/7.create_onnx_resnet.py
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
# Successfully created /home/wudi/work/github/WuChenDi/TensorRT-ONNX/3.read-and-parse-onnx/models/resnet18.onnx

# ==================================================
# Parsing Input Information
# ==================================================
# Input Info:
#   Name:     input0
#   Data Type: FLOAT
#   Shape:     [1, 3, 224, 224]

# ==================================================
# Parsing Output Information
# ==================================================
# Output Info:
#   Name:     output0
#   Data Type: FLOAT
#   Shape:     [1, 1000]

# ==================================================
# Parsing Node Information
# ==================================================
# Node Info:
#   Name:     conv1_conv
#   Op Type:  Conv
#   Inputs:   ['input0', 'conv1_weight', 'conv1_bias']
#   Outputs:  ['conv1_conv_output']
#   Attributes: ['kernel_shape: [7, 7]', 'pads: [3, 3, 3, 3]', 'strides: [2, 2]']
# Node Info:
#   Name:     conv1_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['conv1_conv_output', 'bn1_scale', 'bn1_bias', 'bn1_mean', 'bn1_var']
#   Outputs:  ['conv1_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     conv1_relu
#   Op Type:  Relu
#   Inputs:   ['conv1_bn_output']
#   Outputs:  ['conv1_relu_output']
#   Attributes: []
# Node Info:
#   Name:     maxpool
#   Op Type:  MaxPool
#   Inputs:   ['conv1_relu_output']
#   Outputs:  ['maxpool_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [2, 2]']
# Node Info:
#   Name:     layer1_1_conv1_conv
#   Op Type:  Conv
#   Inputs:   ['maxpool_output', 'layer1_1_conv1_weight', 'layer1_1_conv1_bias']
#   Outputs:  ['layer1_1_conv1_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer1_1_conv1_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer1_1_conv1_conv_output', 'layer1_1_bn1_scale', 'layer1_1_bn1_bias', 'layer1_1_bn1_mean', 'layer1_1_bn1_var']
#   Outputs:  ['layer1_1_conv1_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer1_1_conv1_relu
#   Op Type:  Relu
#   Inputs:   ['layer1_1_conv1_bn_output']
#   Outputs:  ['layer1_1_relu1_output']
#   Attributes: []
# Node Info:
#   Name:     layer1_1_conv2_conv
#   Op Type:  Conv
#   Inputs:   ['layer1_1_relu1_output', 'layer1_1_conv2_weight', 'layer1_1_conv2_bias']
#   Outputs:  ['layer1_1_conv2_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer1_1_conv2_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer1_1_conv2_conv_output', 'layer1_1_bn2_scale', 'layer1_1_bn2_bias', 'layer1_1_bn2_mean', 'layer1_1_bn2_var']
#   Outputs:  ['layer1_1_conv2_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer1_1_conv2_relu
#   Op Type:  Relu
#   Inputs:   ['layer1_1_conv2_bn_output']
#   Outputs:  ['layer1_1_conv2_output']
#   Attributes: []
# Node Info:
#   Name:     layer1_1_add
#   Op Type:  Add
#   Inputs:   ['layer1_1_conv2_output', 'maxpool_output']
#   Outputs:  ['layer1_1_add_output']
#   Attributes: []
# Node Info:
#   Name:     layer1_1_relu
#   Op Type:  Relu
#   Inputs:   ['layer1_1_add_output']
#   Outputs:  ['layer1_1_output']
#   Attributes: []
# Node Info:
#   Name:     layer1_2_conv1_conv
#   Op Type:  Conv
#   Inputs:   ['layer1_1_output', 'layer1_2_conv1_weight', 'layer1_2_conv1_bias']
#   Outputs:  ['layer1_2_conv1_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer1_2_conv1_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer1_2_conv1_conv_output', 'layer1_2_bn1_scale', 'layer1_2_bn1_bias', 'layer1_2_bn1_mean', 'layer1_2_bn1_var']
#   Outputs:  ['layer1_2_conv1_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer1_2_conv1_relu
#   Op Type:  Relu
#   Inputs:   ['layer1_2_conv1_bn_output']
#   Outputs:  ['layer1_2_relu1_output']
#   Attributes: []
# Node Info:
#   Name:     layer1_2_conv2_conv
#   Op Type:  Conv
#   Inputs:   ['layer1_2_relu1_output', 'layer1_2_conv2_weight', 'layer1_2_conv2_bias']
#   Outputs:  ['layer1_2_conv2_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer1_2_conv2_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer1_2_conv2_conv_output', 'layer1_2_bn2_scale', 'layer1_2_bn2_bias', 'layer1_2_bn2_mean', 'layer1_2_bn2_var']
#   Outputs:  ['layer1_2_conv2_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer1_2_conv2_relu
#   Op Type:  Relu
#   Inputs:   ['layer1_2_conv2_bn_output']
#   Outputs:  ['layer1_2_conv2_output']
#   Attributes: []
# Node Info:
#   Name:     layer1_2_add
#   Op Type:  Add
#   Inputs:   ['layer1_2_conv2_output', 'layer1_1_output']
#   Outputs:  ['layer1_2_add_output']
#   Attributes: []
# Node Info:
#   Name:     layer1_2_relu
#   Op Type:  Relu
#   Inputs:   ['layer1_2_add_output']
#   Outputs:  ['layer1_2_output']
#   Attributes: []
# Node Info:
#   Name:     layer2_1_conv1_conv
#   Op Type:  Conv
#   Inputs:   ['layer1_2_output', 'layer2_1_conv1_weight', 'layer2_1_conv1_bias']
#   Outputs:  ['layer2_1_conv1_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [2, 2]']
# Node Info:
#   Name:     layer2_1_conv1_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer2_1_conv1_conv_output', 'layer2_1_bn1_scale', 'layer2_1_bn1_bias', 'layer2_1_bn1_mean', 'layer2_1_bn1_var']
#   Outputs:  ['layer2_1_conv1_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer2_1_conv1_relu
#   Op Type:  Relu
#   Inputs:   ['layer2_1_conv1_bn_output']
#   Outputs:  ['layer2_1_relu1_output']
#   Attributes: []
# Node Info:
#   Name:     layer2_1_conv2_conv
#   Op Type:  Conv
#   Inputs:   ['layer2_1_relu1_output', 'layer2_1_conv2_weight', 'layer2_1_conv2_bias']
#   Outputs:  ['layer2_1_conv2_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer2_1_conv2_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer2_1_conv2_conv_output', 'layer2_1_bn2_scale', 'layer2_1_bn2_bias', 'layer2_1_bn2_mean', 'layer2_1_bn2_var']
#   Outputs:  ['layer2_1_conv2_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer2_1_conv2_relu
#   Op Type:  Relu
#   Inputs:   ['layer2_1_conv2_bn_output']
#   Outputs:  ['layer2_1_conv2_output']
#   Attributes: []
# Node Info:
#   Name:     layer2_1_downsample_conv
#   Op Type:  Conv
#   Inputs:   ['layer1_2_output', 'layer2_1_downsample_conv_weight', 'layer2_1_downsample_conv_bias']
#   Outputs:  ['layer2_1_downsample_conv_output']
#   Attributes: ['kernel_shape: [1, 1]', 'pads: [0, 0, 0, 0]', 'strides: [2, 2]']
# Node Info:
#   Name:     layer2_1_downsample_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer2_1_downsample_conv_output', 'layer2_1_downsample_bn_scale', 'layer2_1_downsample_bn_bias', 'layer2_1_downsample_bn_mean', 'layer2_1_downsample_bn_var']
#   Outputs:  ['layer2_1_downsample_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer2_1_downsample_relu
#   Op Type:  Relu
#   Inputs:   ['layer2_1_downsample_bn_output']
#   Outputs:  ['layer2_1_downsample_output']
#   Attributes: []
# Node Info:
#   Name:     layer2_1_add
#   Op Type:  Add
#   Inputs:   ['layer2_1_conv2_output', 'layer2_1_downsample_output']
#   Outputs:  ['layer2_1_add_output']
#   Attributes: []
# Node Info:
#   Name:     layer2_1_relu
#   Op Type:  Relu
#   Inputs:   ['layer2_1_add_output']
#   Outputs:  ['layer2_1_output']
#   Attributes: []
# Node Info:
#   Name:     layer2_2_conv1_conv
#   Op Type:  Conv
#   Inputs:   ['layer2_1_output', 'layer2_2_conv1_weight', 'layer2_2_conv1_bias']
#   Outputs:  ['layer2_2_conv1_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer2_2_conv1_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer2_2_conv1_conv_output', 'layer2_2_bn1_scale', 'layer2_2_bn1_bias', 'layer2_2_bn1_mean', 'layer2_2_bn1_var']
#   Outputs:  ['layer2_2_conv1_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer2_2_conv1_relu
#   Op Type:  Relu
#   Inputs:   ['layer2_2_conv1_bn_output']
#   Outputs:  ['layer2_2_relu1_output']
#   Attributes: []
# Node Info:
#   Name:     layer2_2_conv2_conv
#   Op Type:  Conv
#   Inputs:   ['layer2_2_relu1_output', 'layer2_2_conv2_weight', 'layer2_2_conv2_bias']
#   Outputs:  ['layer2_2_conv2_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer2_2_conv2_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer2_2_conv2_conv_output', 'layer2_2_bn2_scale', 'layer2_2_bn2_bias', 'layer2_2_bn2_mean', 'layer2_2_bn2_var']
#   Outputs:  ['layer2_2_conv2_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer2_2_conv2_relu
#   Op Type:  Relu
#   Inputs:   ['layer2_2_conv2_bn_output']
#   Outputs:  ['layer2_2_conv2_output']
#   Attributes: []
# Node Info:
#   Name:     layer2_2_add
#   Op Type:  Add
#   Inputs:   ['layer2_2_conv2_output', 'layer2_1_output']
#   Outputs:  ['layer2_2_add_output']
#   Attributes: []
# Node Info:
#   Name:     layer2_2_relu
#   Op Type:  Relu
#   Inputs:   ['layer2_2_add_output']
#   Outputs:  ['layer2_2_output']
#   Attributes: []
# Node Info:
#   Name:     layer3_1_conv1_conv
#   Op Type:  Conv
#   Inputs:   ['layer2_2_output', 'layer3_1_conv1_weight', 'layer3_1_conv1_bias']
#   Outputs:  ['layer3_1_conv1_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [2, 2]']
# Node Info:
#   Name:     layer3_1_conv1_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer3_1_conv1_conv_output', 'layer3_1_bn1_scale', 'layer3_1_bn1_bias', 'layer3_1_bn1_mean', 'layer3_1_bn1_var']
#   Outputs:  ['layer3_1_conv1_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer3_1_conv1_relu
#   Op Type:  Relu
#   Inputs:   ['layer3_1_conv1_bn_output']
#   Outputs:  ['layer3_1_relu1_output']
#   Attributes: []
# Node Info:
#   Name:     layer3_1_conv2_conv
#   Op Type:  Conv
#   Inputs:   ['layer3_1_relu1_output', 'layer3_1_conv2_weight', 'layer3_1_conv2_bias']
#   Outputs:  ['layer3_1_conv2_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer3_1_conv2_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer3_1_conv2_conv_output', 'layer3_1_bn2_scale', 'layer3_1_bn2_bias', 'layer3_1_bn2_mean', 'layer3_1_bn2_var']
#   Outputs:  ['layer3_1_conv2_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer3_1_conv2_relu
#   Op Type:  Relu
#   Inputs:   ['layer3_1_conv2_bn_output']
#   Outputs:  ['layer3_1_conv2_output']
#   Attributes: []
# Node Info:
#   Name:     layer3_1_downsample_conv
#   Op Type:  Conv
#   Inputs:   ['layer2_2_output', 'layer3_1_downsample_conv_weight', 'layer3_1_downsample_conv_bias']
#   Outputs:  ['layer3_1_downsample_conv_output']
#   Attributes: ['kernel_shape: [1, 1]', 'pads: [0, 0, 0, 0]', 'strides: [2, 2]']
# Node Info:
#   Name:     layer3_1_downsample_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer3_1_downsample_conv_output', 'layer3_1_downsample_bn_scale', 'layer3_1_downsample_bn_bias', 'layer3_1_downsample_bn_mean', 'layer3_1_downsample_bn_var']
#   Outputs:  ['layer3_1_downsample_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer3_1_downsample_relu
#   Op Type:  Relu
#   Inputs:   ['layer3_1_downsample_bn_output']
#   Outputs:  ['layer3_1_downsample_output']
#   Attributes: []
# Node Info:
#   Name:     layer3_1_add
#   Op Type:  Add
#   Inputs:   ['layer3_1_conv2_output', 'layer3_1_downsample_output']
#   Outputs:  ['layer3_1_add_output']
#   Attributes: []
# Node Info:
#   Name:     layer3_1_relu
#   Op Type:  Relu
#   Inputs:   ['layer3_1_add_output']
#   Outputs:  ['layer3_1_output']
#   Attributes: []
# Node Info:
#   Name:     layer3_2_conv1_conv
#   Op Type:  Conv
#   Inputs:   ['layer3_1_output', 'layer3_2_conv1_weight', 'layer3_2_conv1_bias']
#   Outputs:  ['layer3_2_conv1_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer3_2_conv1_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer3_2_conv1_conv_output', 'layer3_2_bn1_scale', 'layer3_2_bn1_bias', 'layer3_2_bn1_mean', 'layer3_2_bn1_var']
#   Outputs:  ['layer3_2_conv1_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer3_2_conv1_relu
#   Op Type:  Relu
#   Inputs:   ['layer3_2_conv1_bn_output']
#   Outputs:  ['layer3_2_relu1_output']
#   Attributes: []
# Node Info:
#   Name:     layer3_2_conv2_conv
#   Op Type:  Conv
#   Inputs:   ['layer3_2_relu1_output', 'layer3_2_conv2_weight', 'layer3_2_conv2_bias']
#   Outputs:  ['layer3_2_conv2_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer3_2_conv2_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer3_2_conv2_conv_output', 'layer3_2_bn2_scale', 'layer3_2_bn2_bias', 'layer3_2_bn2_mean', 'layer3_2_bn2_var']
#   Outputs:  ['layer3_2_conv2_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer3_2_conv2_relu
#   Op Type:  Relu
#   Inputs:   ['layer3_2_conv2_bn_output']
#   Outputs:  ['layer3_2_conv2_output']
#   Attributes: []
# Node Info:
#   Name:     layer3_2_add
#   Op Type:  Add
#   Inputs:   ['layer3_2_conv2_output', 'layer3_1_output']
#   Outputs:  ['layer3_2_add_output']
#   Attributes: []
# Node Info:
#   Name:     layer3_2_relu
#   Op Type:  Relu
#   Inputs:   ['layer3_2_add_output']
#   Outputs:  ['layer3_2_output']
#   Attributes: []
# Node Info:
#   Name:     layer4_1_conv1_conv
#   Op Type:  Conv
#   Inputs:   ['layer3_2_output', 'layer4_1_conv1_weight', 'layer4_1_conv1_bias']
#   Outputs:  ['layer4_1_conv1_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [2, 2]']
# Node Info:
#   Name:     layer4_1_conv1_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer4_1_conv1_conv_output', 'layer4_1_bn1_scale', 'layer4_1_bn1_bias', 'layer4_1_bn1_mean', 'layer4_1_bn1_var']
#   Outputs:  ['layer4_1_conv1_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer4_1_conv1_relu
#   Op Type:  Relu
#   Inputs:   ['layer4_1_conv1_bn_output']
#   Outputs:  ['layer4_1_relu1_output']
#   Attributes: []
# Node Info:
#   Name:     layer4_1_conv2_conv
#   Op Type:  Conv
#   Inputs:   ['layer4_1_relu1_output', 'layer4_1_conv2_weight', 'layer4_1_conv2_bias']
#   Outputs:  ['layer4_1_conv2_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer4_1_conv2_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer4_1_conv2_conv_output', 'layer4_1_bn2_scale', 'layer4_1_bn2_bias', 'layer4_1_bn2_mean', 'layer4_1_bn2_var']
#   Outputs:  ['layer4_1_conv2_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer4_1_conv2_relu
#   Op Type:  Relu
#   Inputs:   ['layer4_1_conv2_bn_output']
#   Outputs:  ['layer4_1_conv2_output']
#   Attributes: []
# Node Info:
#   Name:     layer4_1_downsample_conv
#   Op Type:  Conv
#   Inputs:   ['layer3_2_output', 'layer4_1_downsample_conv_weight', 'layer4_1_downsample_conv_bias']
#   Outputs:  ['layer4_1_downsample_conv_output']
#   Attributes: ['kernel_shape: [1, 1]', 'pads: [0, 0, 0, 0]', 'strides: [2, 2]']
# Node Info:
#   Name:     layer4_1_downsample_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer4_1_downsample_conv_output', 'layer4_1_downsample_bn_scale', 'layer4_1_downsample_bn_bias', 'layer4_1_downsample_bn_mean', 'layer4_1_downsample_bn_var']
#   Outputs:  ['layer4_1_downsample_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer4_1_downsample_relu
#   Op Type:  Relu
#   Inputs:   ['layer4_1_downsample_bn_output']
#   Outputs:  ['layer4_1_downsample_output']
#   Attributes: []
# Node Info:
#   Name:     layer4_1_add
#   Op Type:  Add
#   Inputs:   ['layer4_1_conv2_output', 'layer4_1_downsample_output']
#   Outputs:  ['layer4_1_add_output']
#   Attributes: []
# Node Info:
#   Name:     layer4_1_relu
#   Op Type:  Relu
#   Inputs:   ['layer4_1_add_output']
#   Outputs:  ['layer4_1_output']
#   Attributes: []
# Node Info:
#   Name:     layer4_2_conv1_conv
#   Op Type:  Conv
#   Inputs:   ['layer4_1_output', 'layer4_2_conv1_weight', 'layer4_2_conv1_bias']
#   Outputs:  ['layer4_2_conv1_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer4_2_conv1_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer4_2_conv1_conv_output', 'layer4_2_bn1_scale', 'layer4_2_bn1_bias', 'layer4_2_bn1_mean', 'layer4_2_bn1_var']
#   Outputs:  ['layer4_2_conv1_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer4_2_conv1_relu
#   Op Type:  Relu
#   Inputs:   ['layer4_2_conv1_bn_output']
#   Outputs:  ['layer4_2_relu1_output']
#   Attributes: []
# Node Info:
#   Name:     layer4_2_conv2_conv
#   Op Type:  Conv
#   Inputs:   ['layer4_2_relu1_output', 'layer4_2_conv2_weight', 'layer4_2_conv2_bias']
#   Outputs:  ['layer4_2_conv2_conv_output']
#   Attributes: ['kernel_shape: [3, 3]', 'pads: [1, 1, 1, 1]', 'strides: [1, 1]']
# Node Info:
#   Name:     layer4_2_conv2_bn
#   Op Type:  BatchNormalization
#   Inputs:   ['layer4_2_conv2_conv_output', 'layer4_2_bn2_scale', 'layer4_2_bn2_bias', 'layer4_2_bn2_mean', 'layer4_2_bn2_var']
#   Outputs:  ['layer4_2_conv2_bn_output']
#   Attributes: ['epsilon: 9.999999747378752e-06', 'momentum: 0.8999999761581421']
# Node Info:
#   Name:     layer4_2_conv2_relu
#   Op Type:  Relu
#   Inputs:   ['layer4_2_conv2_bn_output']
#   Outputs:  ['layer4_2_conv2_output']
#   Attributes: []
# Node Info:
#   Name:     layer4_2_add
#   Op Type:  Add
#   Inputs:   ['layer4_2_conv2_output', 'layer4_1_output']
#   Outputs:  ['layer4_2_add_output']
#   Attributes: []
# Node Info:
#   Name:     layer4_2_relu
#   Op Type:  Relu
#   Inputs:   ['layer4_2_add_output']
#   Outputs:  ['layer4_2_output']
#   Attributes: []
# Node Info:
#   Name:     avgpool
#   Op Type:  GlobalAveragePool
#   Inputs:   ['layer4_2_output']
#   Outputs:  ['avgpool_output']
#   Attributes: []
# Node Info:
#   Name:     flatten
#   Op Type:  Flatten
#   Inputs:   ['avgpool_output']
#   Outputs:  ['flatten_output']
#   Attributes: ['axis: unsupported']
# Node Info:
#   Name:     fc
#   Op Type:  Gemm
#   Inputs:   ['flatten_output', 'fc_weight', 'fc_bias']
#   Outputs:  ['fc_output']
#   Attributes: ['alpha: 1.0', 'beta: 1.0', 'transB: unsupported']
# Node Info:
#   Name:     output
#   Op Type:  Identity
#   Inputs:   ['fc_output']
#   Outputs:  ['output0']
#   Attributes: []

# ==================================================
# Parsing Initializer Information
# ==================================================
# Initializer Info:
#   Name:     conv1_weight
#   Data Type: FLOAT
#   Shape:     [64, 3, 7, 7]
# Initializer Info:
#   Name:     conv1_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     bn1_scale
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     bn1_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     bn1_mean
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     bn1_var
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_conv1_weight
#   Data Type: FLOAT
#   Shape:     [64, 64, 3, 3]
# Initializer Info:
#   Name:     layer1_1_conv1_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_bn1_scale
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_bn1_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_bn1_mean
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_bn1_var
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_conv2_weight
#   Data Type: FLOAT
#   Shape:     [64, 64, 3, 3]
# Initializer Info:
#   Name:     layer1_1_conv2_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_bn2_scale
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_bn2_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_bn2_mean
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_1_bn2_var
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_conv1_weight
#   Data Type: FLOAT
#   Shape:     [64, 64, 3, 3]
# Initializer Info:
#   Name:     layer1_2_conv1_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_bn1_scale
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_bn1_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_bn1_mean
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_bn1_var
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_conv2_weight
#   Data Type: FLOAT
#   Shape:     [64, 64, 3, 3]
# Initializer Info:
#   Name:     layer1_2_conv2_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_bn2_scale
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_bn2_bias
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_bn2_mean
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer1_2_bn2_var
#   Data Type: FLOAT
#   Shape:     [64]
# Initializer Info:
#   Name:     layer2_1_conv1_weight
#   Data Type: FLOAT
#   Shape:     [128, 64, 3, 3]
# Initializer Info:
#   Name:     layer2_1_conv1_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_bn1_scale
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_bn1_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_bn1_mean
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_bn1_var
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_conv2_weight
#   Data Type: FLOAT
#   Shape:     [128, 128, 3, 3]
# Initializer Info:
#   Name:     layer2_1_conv2_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_bn2_scale
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_bn2_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_bn2_mean
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_bn2_var
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_downsample_conv_weight
#   Data Type: FLOAT
#   Shape:     [128, 64, 1, 1]
# Initializer Info:
#   Name:     layer2_1_downsample_conv_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_downsample_bn_scale
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_downsample_bn_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_downsample_bn_mean
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_1_downsample_bn_var
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_conv1_weight
#   Data Type: FLOAT
#   Shape:     [128, 128, 3, 3]
# Initializer Info:
#   Name:     layer2_2_conv1_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_bn1_scale
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_bn1_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_bn1_mean
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_bn1_var
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_conv2_weight
#   Data Type: FLOAT
#   Shape:     [128, 128, 3, 3]
# Initializer Info:
#   Name:     layer2_2_conv2_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_bn2_scale
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_bn2_bias
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_bn2_mean
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer2_2_bn2_var
#   Data Type: FLOAT
#   Shape:     [128]
# Initializer Info:
#   Name:     layer3_1_conv1_weight
#   Data Type: FLOAT
#   Shape:     [256, 128, 3, 3]
# Initializer Info:
#   Name:     layer3_1_conv1_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_bn1_scale
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_bn1_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_bn1_mean
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_bn1_var
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_conv2_weight
#   Data Type: FLOAT
#   Shape:     [256, 256, 3, 3]
# Initializer Info:
#   Name:     layer3_1_conv2_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_bn2_scale
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_bn2_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_bn2_mean
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_bn2_var
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_downsample_conv_weight
#   Data Type: FLOAT
#   Shape:     [256, 128, 1, 1]
# Initializer Info:
#   Name:     layer3_1_downsample_conv_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_downsample_bn_scale
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_downsample_bn_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_downsample_bn_mean
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_1_downsample_bn_var
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_conv1_weight
#   Data Type: FLOAT
#   Shape:     [256, 256, 3, 3]
# Initializer Info:
#   Name:     layer3_2_conv1_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_bn1_scale
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_bn1_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_bn1_mean
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_bn1_var
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_conv2_weight
#   Data Type: FLOAT
#   Shape:     [256, 256, 3, 3]
# Initializer Info:
#   Name:     layer3_2_conv2_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_bn2_scale
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_bn2_bias
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_bn2_mean
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer3_2_bn2_var
#   Data Type: FLOAT
#   Shape:     [256]
# Initializer Info:
#   Name:     layer4_1_conv1_weight
#   Data Type: FLOAT
#   Shape:     [512, 256, 3, 3]
# Initializer Info:
#   Name:     layer4_1_conv1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_bn1_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_bn1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_bn1_mean
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_bn1_var
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_conv2_weight
#   Data Type: FLOAT
#   Shape:     [512, 512, 3, 3]
# Initializer Info:
#   Name:     layer4_1_conv2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_bn2_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_bn2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_bn2_mean
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_bn2_var
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_downsample_conv_weight
#   Data Type: FLOAT
#   Shape:     [512, 256, 1, 1]
# Initializer Info:
#   Name:     layer4_1_downsample_conv_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_downsample_bn_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_downsample_bn_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_downsample_bn_mean
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_1_downsample_bn_var
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_conv1_weight
#   Data Type: FLOAT
#   Shape:     [512, 512, 3, 3]
# Initializer Info:
#   Name:     layer4_2_conv1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_bn1_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_bn1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_bn1_mean
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_bn1_var
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_conv2_weight
#   Data Type: FLOAT
#   Shape:     [512, 512, 3, 3]
# Initializer Info:
#   Name:     layer4_2_conv2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_bn2_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_bn2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_bn2_mean
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     layer4_2_bn2_var
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer Info:
#   Name:     fc_weight
#   Data Type: FLOAT
#   Shape:     [1000, 512]
# Initializer Info:
#   Name:     fc_bias
#   Data Type: FLOAT
#   Shape:     [1000]
