import numpy as np
import onnx
from onnx import helper, TensorProto
from utils import get_onnx_path

def create_initializer_tensor(
    name: str,
    tensor_array: np.ndarray,
    data_type: int = TensorProto.FLOAT
) -> onnx.TensorProto:
    """Create an ONNX TensorProto initializer from a numpy array.

    Args:
        name: Name of the initializer.
        tensor_array: Numpy array containing the tensor data.
        data_type: ONNX data type (e.g., TensorProto.FLOAT).

    Returns:
        TensorProto: ONNX initializer tensor.
    """
    initializer = helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist()
    )
    return initializer

def parse_onnx(model: onnx.ModelProto) -> None:
    """Parse and print ONNX model structure.

    Args:
        model: The ONNX model to parse.

    Note:
        - Parses inputs, outputs, nodes, and initializers from model.graph.
        - Prints name, data type, shape, and node attributes.
    """
    try:
        graph = model.graph
        if not graph:
            raise ValueError("Model graph is empty or invalid")

        print(f"\n{'='*50}")
        print("Parsing Input Information")
        print(f"{'='*50}")
        for input in graph.input:
            input_shape = [d.dim_value if d.dim_value != 0 else None for d in input.type.tensor_type.shape.dim]
            print(f"Input info:\n"
                  f"  Name:      {input.name}\n"
                  f"  Data Type: {TensorProto.DataType.Name(input.type.tensor_type.elem_type)}\n"
                  f"  Shape:     {input_shape}")

        print(f"\n{'='*50}")
        print("Parsing Output Information")
        print(f"{'='*50}")
        for output in graph.output:
            output_shape = [d.dim_value if d.dim_value != 0 else None for d in output.type.tensor_type.shape.dim]
            print(f"Output info:\n"
                  f"  Name:      {output.name}\n"
                  f"  Data Type: {TensorProto.DataType.Name(output.type.tensor_type.elem_type)}\n"
                  f"  Shape:     {output_shape}")

        print(f"\n{'='*50}")
        print("Parsing Node Information")
        print(f"{'='*50}")
        for node in graph.node:
            attributes = [f"{attr.name}: {attr.ints or attr.floats or attr.s.decode() or attr.f}" for attr in node.attribute]
            print(f"Node info:\n"
                  f"  Name:      {node.name}\n"
                  f"  Op Type:   {node.op_type}\n"
                  f"  Inputs:    {node.input}\n"
                  f"  Outputs:   {node.output}\n"
                  f"  Attributes: {attributes}")

        print(f"\n{'='*50}")
        print("Parsing Initializer Information")
        print(f"{'='*50}")
        for initializer in graph.initializer:
            print(f"Initializer info:\n"
                  f"  Name:      {initializer.name}\n"
                  f"  Data Type: {TensorProto.DataType.Name(initializer.data_type)}\n"
                  f"  Shape:     {list(initializer.dims)}")
    except Exception as e:
        print(f"Failed to parse model: {str(e)}")

def main():
    """Create and save a Transformer ONNX model."""
    # Model configuration
    batch_size = 1
    seq_length = 128
    input_dim = 512
    num_heads = 8
    head_dim = input_dim // num_heads  # 64
    hidden_size = 512
    feedforward_size = 2048
    num_layers = 6

    input_shape = [batch_size, seq_length, input_dim]
    output_shape = [batch_size, seq_length, input_dim]

    # Create input and output
    model_input_name = "input0"
    model_output_name = "output0"

    input = helper.make_tensor_value_info(
        model_input_name, TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(
        model_output_name, TensorProto.FLOAT, output_shape)

    nodes = []
    initializers = []
    previous_output_name = model_input_name

    # Create Transformer layers
    for layer in range(num_layers):
        layer_norm1_output = f"layer{layer}_norm1.output"
        attn_output = f"layer{layer}_attn.output"
        attn_output_add = f"layer{layer}_attn_add.output"
        layer_norm2_output = f"layer{layer}_norm2.output"
        ff_output = f"layer{layer}_ff.output"
        ff_output_add = f"layer{layer}_ff_add.output"

        # Layer Normalization 1
        ln1_scale = np.ones(input_dim, dtype=np.float32)
        ln1_bias = np.zeros(input_dim, dtype=np.float32)
        ln1_scale_initializer = create_initializer_tensor(
            f"layer{layer}_ln1_scale", ln1_scale, TensorProto.FLOAT)
        ln1_bias_initializer = create_initializer_tensor(
            f"layer{layer}_ln1_bias", ln1_bias, TensorProto.FLOAT)
        initializers.extend([ln1_scale_initializer, ln1_bias_initializer])

        ln1_node = helper.make_node(
            "LayerNormalization",
            inputs=[previous_output_name, f"layer{layer}_ln1_scale", f"layer{layer}_ln1_bias"],
            outputs=[layer_norm1_output],
            name=f"layer{layer}_ln1",
            epsilon=1e-5
        )
        nodes.append(ln1_node)

        # Multi-Head Attention
        qkv_weight = np.random.randn(3 * input_dim, input_dim).astype(np.float32) * np.sqrt(2.0 / (input_dim + 3 * input_dim))
        qkv_bias = np.random.randn(3 * input_dim).astype(np.float32) * 0.01
        # print(f"Layer {layer} QKV weight shape: {qkv_weight.shape}, bias shape: {qkv_bias.shape}")
        qkv_weight_initializer = create_initializer_tensor(
            f"layer{layer}_qkv_weight", qkv_weight, TensorProto.FLOAT)
        qkv_bias_initializer = create_initializer_tensor(
            f"layer{layer}_qkv_bias", qkv_bias, TensorProto.FLOAT)
        initializers.extend([qkv_weight_initializer, qkv_bias_initializer])

        proj_weight = np.random.randn(input_dim, input_dim).astype(np.float32) * np.sqrt(2.0 / (input_dim + input_dim))
        proj_bias = np.random.randn(input_dim).astype(np.float32) * 0.01
        proj_weight_initializer = create_initializer_tensor(
            f"layer{layer}_proj_weight", proj_weight, TensorProto.FLOAT)
        proj_bias_initializer = create_initializer_tensor(
            f"layer{layer}_proj_bias", proj_bias, TensorProto.FLOAT)
        initializers.extend([proj_weight_initializer, proj_bias_initializer])

        # QKV computation
        qkv_matmul_output = f"layer{layer}_qkv_matmul.output"
        qkv_matmul_node = helper.make_node(
            "MatMul",
            inputs=[layer_norm1_output, f"layer{layer}_qkv_weight"],
            outputs=[qkv_matmul_output],
            name=f"layer{layer}_qkv_matmul"
        )
        nodes.append(qkv_matmul_node)

        qkv_bias_add_output = f"layer{layer}_qkv_bias_add.output"
        qkv_bias_add_node = helper.make_node(
            "Add",
            inputs=[qkv_matmul_output, f"layer{layer}_qkv_bias"],
            outputs=[qkv_bias_add_output],
            name=f"layer{layer}_qkv_bias_add"
        )
        nodes.append(qkv_bias_add_node)

        # Split Q, K, V
        q_output = f"layer{layer}_q.output"
        k_output = f"layer{layer}_k.output"
        v_output = f"layer{layer}_v.output"

        split_outputs = [q_output, k_output, v_output]
        # split_node = helper.make_node(
        #     "Split",
        #     inputs=[qkv_bias_add_output],
        #     outputs=split_outputs,
        #     name=f"layer{layer}_qkv_split",
        #     axis=-1,
        #     split=[input_dim, input_dim, hidden_size]
        # )
        split_node = helper.make_node(
            "Split",
            inputs=[qkv_bias_add_output],
            outputs=split_outputs,
            name=f"layer{layer}_qkv_split",
            axis=-1
        )
        nodes.append(split_node)

        # Reshape for heads
        def reshape_for_heads(name, input_name):
            reshape_output = f"{name}_reshape.output"
            transpose_output = f"{name}_transpose.output"
            shape = [batch_size, seq_length, num_heads, head_dim]
            shape_initializer = create_initializer_tensor(
                f"{name}_shape", np.array(shape, dtype=np.int64), TensorProto.INT64)
            initializers.append(shape_initializer)
            reshape_node = helper.make_node(
                "Reshape",
                inputs=[input_name, f"{name}_shape"],
                outputs=[reshape_output],
                name=f"{name}_reshape"
            )
            transpose_node = helper.make_node(
                "Transpose",
                inputs=[reshape_output],
                outputs=[transpose_output],
                name=f"{name}_transpose",
                perm=[0, 2, 1, 3]
            )
            return [reshape_node, transpose_node], transpose_output

        q_nodes, q_reshaped = reshape_for_heads(f"layer{layer}_q", q_output)
        k_nodes, k_reshaped = reshape_for_heads(f"layer{layer}_k", k_output)
        v_nodes, v_reshaped = reshape_for_heads(f"layer{layer}_v", v_output)
        nodes.extend(q_nodes + k_nodes + v_nodes)

        # Attention computation
        attn_matmul_qk_output = f"layer{layer}_attn_matmul_qk.output"
        attn_matmul_qk_node = helper.make_node(
            "MatMul",
            inputs=[q_reshaped, k_reshaped],
            outputs=[attn_matmul_qk_output],
            name=f"layer{layer}_attn_matmul_qk"
        )
        nodes.append(attn_matmul_qk_node)

        scale_value = np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32)
        scale_initializer = create_initializer_tensor(
            f"layer{layer}_scale", scale_value, TensorProto.FLOAT)
        initializers.append(scale_initializer)

        attn_scaled_output = f"layer{layer}_attn_scaled.output"
        attn_scale_node = helper.make_node(
            "Mul",
            inputs=[attn_matmul_qk_output, f"layer{layer}_scale"],
            outputs=[attn_scaled_output],
            name=f"layer{layer}_attn_scale"
        )
        nodes.append(attn_scale_node)

        attn_softmax_output = f"layer{layer}_attn_softmax.output"
        attn_softmax_node = helper.make_node(
            "Softmax",
            inputs=[attn_scaled_output],
            outputs=[attn_softmax_output],
            name=f"layer{layer}_attn_softmax",
            axis=1
        )
        nodes.append(attn_softmax_node)

        attn_matmul_v_output = f"layer{layer}_attn_matmul_v.output"
        attn_matmul_v_node = helper.make_node(
            "MatMul",
            inputs=[attn_softmax_output, v_reshaped],
            outputs=[attn_matmul_v_output],
            name=f"layer{layer}_attn_matmul_v"
        )
        nodes.append(attn_matmul_v_node)

        attn_transpose_output = f"layer{layer}_attn_transpose.output"
        attn_transpose_node = helper.make_node(
            "Transpose",
            inputs=[attn_matmul_v_output],
            outputs=[attn_transpose_output],
            name=f"layer{layer}_attn_transpose",
            perm=[0, 2, 1, 3]
        )
        nodes.append(attn_transpose_node)

        attn_reshape_output = f"layer{layer}_attn_reshape.output"
        attn_reshape_shape = np.array([batch_size, seq_length, input_dim], dtype=np.int64)
        attn_reshape_shape_initializer = create_initializer_tensor(
            f"layer{layer}_attn_reshape_shape", attn_reshape_shape, TensorProto.INT64)
        initializers.append(attn_reshape_shape_initializer)

        attn_reshape_node = helper.make_node(
            "Reshape",
            inputs=[attn_transpose_output, f"layer{layer}_attn_reshape_shape"],
            outputs=[attn_reshape_output],
            name=f"layer{layer}_attn_reshape"
        )
        nodes.append(attn_reshape_node)

        # Output projection
        proj_matmul_output = f"layer{layer}_proj_matmul.output"
        proj_matmul_node = helper.make_node(
            "MatMul",
            inputs=[attn_reshape_output, f"layer{layer}_proj_weight"],
            outputs=[proj_matmul_output],
            name=f"layer{layer}_proj_matmul"
        )
        nodes.append(proj_matmul_node)

        proj_bias_add_output = f"layer{layer}_proj_bias_add.output"
        proj_bias_add_node = helper.make_node(
            "Add",
            inputs=[proj_matmul_output, f"layer{layer}_proj_bias"],
            outputs=[proj_bias_add_output],
            name=f"layer{layer}_proj_bias_add"
        )
        nodes.append(proj_bias_add_node)

        # Residual connection 1
        attn_output_add_node = helper.make_node(
            "Add",
            inputs=[previous_output_name, proj_bias_add_output],
            outputs=[attn_output_add],
            name=f"layer{layer}_attn_add"
        )
        nodes.append(attn_output_add_node)

        # Layer Normalization 2
        ln2_scale = np.ones(input_dim, dtype=np.float32)
        ln2_bias = np.zeros(input_dim, dtype=np.float32)
        ln2_scale_initializer = create_initializer_tensor(
            f"layer{layer}_ln2_scale", ln2_scale, TensorProto.FLOAT)
        ln2_bias_initializer = create_initializer_tensor(
            f"layer{layer}_ln2_bias", ln2_bias, TensorProto.FLOAT)
        initializers.extend([ln2_scale_initializer, ln2_bias_initializer])

        ln2_node = helper.make_node(
            "LayerNormalization",
            inputs=[attn_output_add, f"layer{layer}_ln2_scale", f"layer{layer}_ln2_bias"],
            outputs=[layer_norm2_output],
            name=f"layer{layer}_norm2",
            epsilon=1e-5
        )
        nodes.append(ln2_node)

        # Feedforward projection
        ff_weight1 = np.random.randn(feedforward_size, input_dim).astype(np.float32) * np.sqrt(2.0 / (feedforward_size + input_dim))
        ff_bias1 = np.zeros(feedforward_size, dtype=np.float32)
        ff_weight1_initializer = create_initializer_tensor(
            f"layer{layer}_ff_weight1", ff_weight1, TensorProto.FLOAT)
        ff_bias1_initializer = create_initializer_tensor(
            f"layer{layer}_ff_bias1", ff_bias1, TensorProto.FLOAT)
        initializers.extend([ff_weight1_initializer, ff_bias1_initializer])

        ff_matmul1_output = f"layer{layer}_ff_matmul1.output"
        ff_matmul1_node = helper.make_node(
            "MatMul",
            inputs=[layer_norm2_output, f"layer{layer}_ff_weight1"],
            outputs=[ff_matmul1_output],
            name=f"layer{layer}_ff_matmul1"
        )
        nodes.append(ff_matmul1_node)

        ff_bias1_add_output = f"layer{layer}_ff_bias1_add.output"
        ff_bias1_add_node = helper.make_node(
            "Add",
            inputs=[ff_matmul1_output, f"layer{layer}_ff_bias1"],
            outputs=[ff_bias1_add_output],
            name=f"layer{layer}_ff_bias1_add"
        )
        nodes.append(ff_bias1_add_node)

        ff_relu_output = f"layer{layer}_ff_relu.output"
        ff_relu_node = helper.make_node(
            "Relu",
            inputs=[ff_bias1_add_output],
            outputs=[ff_relu_output],
            name=f"layer{layer}_ff_relu"
        )
        nodes.append(ff_relu_node)

        ff_weight2 = np.random.randn(input_dim, feedforward_size).astype(np.float32) * np.sqrt(2.0 / (input_dim + feedforward_size))
        ff_bias2 = np.zeros(input_dim, dtype=np.float32)
        ff_weight2_initializer = create_initializer_tensor(
            f"layer{layer}_ff_weight2", ff_weight2, TensorProto.FLOAT)
        ff_bias2_initializer = create_initializer_tensor(
            f"layer{layer}_ff_bias2", ff_bias2, TensorProto.FLOAT)
        initializers.extend([ff_weight2_initializer, ff_bias2_initializer])

        ff_matmul2_output = f"layer{layer}_ff_matmul2.output"
        ff_matmul2_node = helper.make_node(
            "MatMul",
            inputs=[ff_relu_output, f"layer{layer}_ff_weight2"],
            outputs=[ff_matmul2_output],
            name=f"layer{layer}_ff_matmul2"
        )
        nodes.append(ff_matmul2_node)

        ff_bias2_add_output = f"layer{layer}_ff_bias2_add.output"
        ff_bias2_add_node = helper.make_node(
            "Add",
            inputs=[ff_matmul2_output, f"layer{layer}_ff_bias2"],
            outputs=[ff_bias2_add_output],
            name=f"layer{layer}_ff_bias2_add"
        )
        nodes.append(ff_bias2_add_node)

        # Residual connection 2
        ff_output_add_node = helper.make_node(
            "Add",
            inputs=[attn_output_add, ff_bias2_add_output],
            outputs=[ff_output_add],
            name=f"layer{layer}_ff_add"
        )
        nodes.append(ff_output_add_node)

        previous_output_name = ff_output_add

    # Final LayerNormalization
    ln_final_scale = np.ones(input_dim, dtype=np.float32)
    ln_final_bias = np.zeros(input_dim, dtype=np.float32)
    ln_final_scale_initializer = create_initializer_tensor(
        "ln_final_scale", ln_final_scale, TensorProto.FLOAT)
    ln_final_bias_initializer = create_initializer_tensor(
        "ln_final_bias", ln_final_bias, TensorProto.FLOAT)
    initializers.extend([ln_final_scale_initializer, ln_final_bias_initializer])

    ln_final_node = helper.make_node(
        "LayerNormalization",
        inputs=[previous_output_name, "ln_final_scale", "ln_final_bias"],
        outputs=[model_output_name],
        name="ln_final",
        epsilon=1e-5
    )
    nodes.append(ln_final_node)

    # Create graph
    graph = helper.make_graph(
        nodes=nodes,
        name="transformer",
        inputs=[input],
        outputs=[output],
        initializer=initializers
    )

    # Create model
    model = helper.make_model(graph, producer_name="onnx-transformer-sample")
    model.opset_import[0].version = 17

    # Validate and save model
    try:
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        output_path = get_onnx_path(__file__, "transformer.onnx")
        onnx.save(model, output_path)
        print(f"Successfully created model: {output_path}")

        # Parse model for debugging
        parse_onnx(model)
        print(f"\nTotal nodes: {len(nodes)}")
        print(f"Total initializers: {len(initializers)}")
    except Exception as e:
        print(f"Failed to create or save model: {str(e)}")

if __name__ == "__main__":
    main()

# python3 3.read-and-parse-onnx/6.create_onnx_transformer.py
# Successfully created model: /home/wudi/work/github/WuChenDi/TensorRT-ONNX/3.read-and-parse-onnx/models/transformer.onnx

# ==================================================
# Parsing Input Information
# ==================================================
# Input info:
#   Name:      input0
#   Data Type: FLOAT
#   Shape:     [1, 128, 512]

# ==================================================
# Parsing Output Information
# ==================================================
# Output info:
#   Name:      output0
#   Data Type: FLOAT
#   Shape:     [1, 128, 512]

# ==================================================
# Parsing Node Information
# ==================================================
# Node info:
#   Name:      layer0_ln1
#   Op Type:   LayerNormalization
#   Inputs:    ['input0', 'layer0_ln1_scale', 'layer0_ln1_bias']
#   Outputs:   ['layer0_norm1.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer0_qkv_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer0_norm1.output', 'layer0_qkv_weight']
#   Outputs:   ['layer0_qkv_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer0_qkv_bias_add
#   Op Type:   Add
#   Inputs:    ['layer0_qkv_matmul.output', 'layer0_qkv_bias']
#   Outputs:   ['layer0_qkv_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer0_qkv_split
#   Op Type:   Split
#   Inputs:    ['layer0_qkv_bias_add.output']
#   Outputs:   ['layer0_q.output', 'layer0_k.output', 'layer0_v.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer0_q_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer0_q.output', 'layer0_q_shape']
#   Outputs:   ['layer0_q_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer0_q_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer0_q_reshape.output']
#   Outputs:   ['layer0_q_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer0_k_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer0_k.output', 'layer0_k_shape']
#   Outputs:   ['layer0_k_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer0_k_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer0_k_reshape.output']
#   Outputs:   ['layer0_k_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer0_v_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer0_v.output', 'layer0_v_shape']
#   Outputs:   ['layer0_v_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer0_v_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer0_v_reshape.output']
#   Outputs:   ['layer0_v_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer0_attn_matmul_qk
#   Op Type:   MatMul
#   Inputs:    ['layer0_q_transpose.output', 'layer0_k_transpose.output']
#   Outputs:   ['layer0_attn_matmul_qk.output']
#   Attributes: []
# Node info:
#   Name:      layer0_attn_scale
#   Op Type:   Mul
#   Inputs:    ['layer0_attn_matmul_qk.output', 'layer0_scale']
#   Outputs:   ['layer0_attn_scaled.output']
#   Attributes: []
# Node info:
#   Name:      layer0_attn_softmax
#   Op Type:   Softmax
#   Inputs:    ['layer0_attn_scaled.output']
#   Outputs:   ['layer0_attn_softmax.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer0_attn_matmul_v
#   Op Type:   MatMul
#   Inputs:    ['layer0_attn_softmax.output', 'layer0_v_transpose.output']
#   Outputs:   ['layer0_attn_matmul_v.output']
#   Attributes: []
# Node info:
#   Name:      layer0_attn_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer0_attn_matmul_v.output']
#   Outputs:   ['layer0_attn_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer0_attn_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer0_attn_transpose.output', 'layer0_attn_reshape_shape']
#   Outputs:   ['layer0_attn_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer0_proj_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer0_attn_reshape.output', 'layer0_proj_weight']
#   Outputs:   ['layer0_proj_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer0_proj_bias_add
#   Op Type:   Add
#   Inputs:    ['layer0_proj_matmul.output', 'layer0_proj_bias']
#   Outputs:   ['layer0_proj_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer0_attn_add
#   Op Type:   Add
#   Inputs:    ['input0', 'layer0_proj_bias_add.output']
#   Outputs:   ['layer0_attn_add.output']
#   Attributes: []
# Node info:
#   Name:      layer0_norm2
#   Op Type:   LayerNormalization
#   Inputs:    ['layer0_attn_add.output', 'layer0_ln2_scale', 'layer0_ln2_bias']
#   Outputs:   ['layer0_norm2.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer0_ff_matmul1
#   Op Type:   MatMul
#   Inputs:    ['layer0_norm2.output', 'layer0_ff_weight1']
#   Outputs:   ['layer0_ff_matmul1.output']
#   Attributes: []
# Node info:
#   Name:      layer0_ff_bias1_add
#   Op Type:   Add
#   Inputs:    ['layer0_ff_matmul1.output', 'layer0_ff_bias1']
#   Outputs:   ['layer0_ff_bias1_add.output']
#   Attributes: []
# Node info:
#   Name:      layer0_ff_relu
#   Op Type:   Relu
#   Inputs:    ['layer0_ff_bias1_add.output']
#   Outputs:   ['layer0_ff_relu.output']
#   Attributes: []
# Node info:
#   Name:      layer0_ff_matmul2
#   Op Type:   MatMul
#   Inputs:    ['layer0_ff_relu.output', 'layer0_ff_weight2']
#   Outputs:   ['layer0_ff_matmul2.output']
#   Attributes: []
# Node info:
#   Name:      layer0_ff_bias2_add
#   Op Type:   Add
#   Inputs:    ['layer0_ff_matmul2.output', 'layer0_ff_bias2']
#   Outputs:   ['layer0_ff_bias2_add.output']
#   Attributes: []
# Node info:
#   Name:      layer0_ff_add
#   Op Type:   Add
#   Inputs:    ['layer0_attn_add.output', 'layer0_ff_bias2_add.output']
#   Outputs:   ['layer0_ff_add.output']
#   Attributes: []
# Node info:
#   Name:      layer1_ln1
#   Op Type:   LayerNormalization
#   Inputs:    ['layer0_ff_add.output', 'layer1_ln1_scale', 'layer1_ln1_bias']
#   Outputs:   ['layer1_norm1.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer1_qkv_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer1_norm1.output', 'layer1_qkv_weight']
#   Outputs:   ['layer1_qkv_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer1_qkv_bias_add
#   Op Type:   Add
#   Inputs:    ['layer1_qkv_matmul.output', 'layer1_qkv_bias']
#   Outputs:   ['layer1_qkv_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer1_qkv_split
#   Op Type:   Split
#   Inputs:    ['layer1_qkv_bias_add.output']
#   Outputs:   ['layer1_q.output', 'layer1_k.output', 'layer1_v.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer1_q_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer1_q.output', 'layer1_q_shape']
#   Outputs:   ['layer1_q_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer1_q_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer1_q_reshape.output']
#   Outputs:   ['layer1_q_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer1_k_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer1_k.output', 'layer1_k_shape']
#   Outputs:   ['layer1_k_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer1_k_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer1_k_reshape.output']
#   Outputs:   ['layer1_k_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer1_v_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer1_v.output', 'layer1_v_shape']
#   Outputs:   ['layer1_v_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer1_v_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer1_v_reshape.output']
#   Outputs:   ['layer1_v_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer1_attn_matmul_qk
#   Op Type:   MatMul
#   Inputs:    ['layer1_q_transpose.output', 'layer1_k_transpose.output']
#   Outputs:   ['layer1_attn_matmul_qk.output']
#   Attributes: []
# Node info:
#   Name:      layer1_attn_scale
#   Op Type:   Mul
#   Inputs:    ['layer1_attn_matmul_qk.output', 'layer1_scale']
#   Outputs:   ['layer1_attn_scaled.output']
#   Attributes: []
# Node info:
#   Name:      layer1_attn_softmax
#   Op Type:   Softmax
#   Inputs:    ['layer1_attn_scaled.output']
#   Outputs:   ['layer1_attn_softmax.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer1_attn_matmul_v
#   Op Type:   MatMul
#   Inputs:    ['layer1_attn_softmax.output', 'layer1_v_transpose.output']
#   Outputs:   ['layer1_attn_matmul_v.output']
#   Attributes: []
# Node info:
#   Name:      layer1_attn_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer1_attn_matmul_v.output']
#   Outputs:   ['layer1_attn_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer1_attn_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer1_attn_transpose.output', 'layer1_attn_reshape_shape']
#   Outputs:   ['layer1_attn_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer1_proj_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer1_attn_reshape.output', 'layer1_proj_weight']
#   Outputs:   ['layer1_proj_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer1_proj_bias_add
#   Op Type:   Add
#   Inputs:    ['layer1_proj_matmul.output', 'layer1_proj_bias']
#   Outputs:   ['layer1_proj_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer1_attn_add
#   Op Type:   Add
#   Inputs:    ['layer0_ff_add.output', 'layer1_proj_bias_add.output']
#   Outputs:   ['layer1_attn_add.output']
#   Attributes: []
# Node info:
#   Name:      layer1_norm2
#   Op Type:   LayerNormalization
#   Inputs:    ['layer1_attn_add.output', 'layer1_ln2_scale', 'layer1_ln2_bias']
#   Outputs:   ['layer1_norm2.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer1_ff_matmul1
#   Op Type:   MatMul
#   Inputs:    ['layer1_norm2.output', 'layer1_ff_weight1']
#   Outputs:   ['layer1_ff_matmul1.output']
#   Attributes: []
# Node info:
#   Name:      layer1_ff_bias1_add
#   Op Type:   Add
#   Inputs:    ['layer1_ff_matmul1.output', 'layer1_ff_bias1']
#   Outputs:   ['layer1_ff_bias1_add.output']
#   Attributes: []
# Node info:
#   Name:      layer1_ff_relu
#   Op Type:   Relu
#   Inputs:    ['layer1_ff_bias1_add.output']
#   Outputs:   ['layer1_ff_relu.output']
#   Attributes: []
# Node info:
#   Name:      layer1_ff_matmul2
#   Op Type:   MatMul
#   Inputs:    ['layer1_ff_relu.output', 'layer1_ff_weight2']
#   Outputs:   ['layer1_ff_matmul2.output']
#   Attributes: []
# Node info:
#   Name:      layer1_ff_bias2_add
#   Op Type:   Add
#   Inputs:    ['layer1_ff_matmul2.output', 'layer1_ff_bias2']
#   Outputs:   ['layer1_ff_bias2_add.output']
#   Attributes: []
# Node info:
#   Name:      layer1_ff_add
#   Op Type:   Add
#   Inputs:    ['layer1_attn_add.output', 'layer1_ff_bias2_add.output']
#   Outputs:   ['layer1_ff_add.output']
#   Attributes: []
# Node info:
#   Name:      layer2_ln1
#   Op Type:   LayerNormalization
#   Inputs:    ['layer1_ff_add.output', 'layer2_ln1_scale', 'layer2_ln1_bias']
#   Outputs:   ['layer2_norm1.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer2_qkv_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer2_norm1.output', 'layer2_qkv_weight']
#   Outputs:   ['layer2_qkv_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer2_qkv_bias_add
#   Op Type:   Add
#   Inputs:    ['layer2_qkv_matmul.output', 'layer2_qkv_bias']
#   Outputs:   ['layer2_qkv_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer2_qkv_split
#   Op Type:   Split
#   Inputs:    ['layer2_qkv_bias_add.output']
#   Outputs:   ['layer2_q.output', 'layer2_k.output', 'layer2_v.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer2_q_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer2_q.output', 'layer2_q_shape']
#   Outputs:   ['layer2_q_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer2_q_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer2_q_reshape.output']
#   Outputs:   ['layer2_q_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer2_k_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer2_k.output', 'layer2_k_shape']
#   Outputs:   ['layer2_k_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer2_k_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer2_k_reshape.output']
#   Outputs:   ['layer2_k_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer2_v_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer2_v.output', 'layer2_v_shape']
#   Outputs:   ['layer2_v_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer2_v_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer2_v_reshape.output']
#   Outputs:   ['layer2_v_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer2_attn_matmul_qk
#   Op Type:   MatMul
#   Inputs:    ['layer2_q_transpose.output', 'layer2_k_transpose.output']
#   Outputs:   ['layer2_attn_matmul_qk.output']
#   Attributes: []
# Node info:
#   Name:      layer2_attn_scale
#   Op Type:   Mul
#   Inputs:    ['layer2_attn_matmul_qk.output', 'layer2_scale']
#   Outputs:   ['layer2_attn_scaled.output']
#   Attributes: []
# Node info:
#   Name:      layer2_attn_softmax
#   Op Type:   Softmax
#   Inputs:    ['layer2_attn_scaled.output']
#   Outputs:   ['layer2_attn_softmax.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer2_attn_matmul_v
#   Op Type:   MatMul
#   Inputs:    ['layer2_attn_softmax.output', 'layer2_v_transpose.output']
#   Outputs:   ['layer2_attn_matmul_v.output']
#   Attributes: []
# Node info:
#   Name:      layer2_attn_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer2_attn_matmul_v.output']
#   Outputs:   ['layer2_attn_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer2_attn_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer2_attn_transpose.output', 'layer2_attn_reshape_shape']
#   Outputs:   ['layer2_attn_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer2_proj_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer2_attn_reshape.output', 'layer2_proj_weight']
#   Outputs:   ['layer2_proj_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer2_proj_bias_add
#   Op Type:   Add
#   Inputs:    ['layer2_proj_matmul.output', 'layer2_proj_bias']
#   Outputs:   ['layer2_proj_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer2_attn_add
#   Op Type:   Add
#   Inputs:    ['layer1_ff_add.output', 'layer2_proj_bias_add.output']
#   Outputs:   ['layer2_attn_add.output']
#   Attributes: []
# Node info:
#   Name:      layer2_norm2
#   Op Type:   LayerNormalization
#   Inputs:    ['layer2_attn_add.output', 'layer2_ln2_scale', 'layer2_ln2_bias']
#   Outputs:   ['layer2_norm2.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer2_ff_matmul1
#   Op Type:   MatMul
#   Inputs:    ['layer2_norm2.output', 'layer2_ff_weight1']
#   Outputs:   ['layer2_ff_matmul1.output']
#   Attributes: []
# Node info:
#   Name:      layer2_ff_bias1_add
#   Op Type:   Add
#   Inputs:    ['layer2_ff_matmul1.output', 'layer2_ff_bias1']
#   Outputs:   ['layer2_ff_bias1_add.output']
#   Attributes: []
# Node info:
#   Name:      layer2_ff_relu
#   Op Type:   Relu
#   Inputs:    ['layer2_ff_bias1_add.output']
#   Outputs:   ['layer2_ff_relu.output']
#   Attributes: []
# Node info:
#   Name:      layer2_ff_matmul2
#   Op Type:   MatMul
#   Inputs:    ['layer2_ff_relu.output', 'layer2_ff_weight2']
#   Outputs:   ['layer2_ff_matmul2.output']
#   Attributes: []
# Node info:
#   Name:      layer2_ff_bias2_add
#   Op Type:   Add
#   Inputs:    ['layer2_ff_matmul2.output', 'layer2_ff_bias2']
#   Outputs:   ['layer2_ff_bias2_add.output']
#   Attributes: []
# Node info:
#   Name:      layer2_ff_add
#   Op Type:   Add
#   Inputs:    ['layer2_attn_add.output', 'layer2_ff_bias2_add.output']
#   Outputs:   ['layer2_ff_add.output']
#   Attributes: []
# Node info:
#   Name:      layer3_ln1
#   Op Type:   LayerNormalization
#   Inputs:    ['layer2_ff_add.output', 'layer3_ln1_scale', 'layer3_ln1_bias']
#   Outputs:   ['layer3_norm1.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer3_qkv_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer3_norm1.output', 'layer3_qkv_weight']
#   Outputs:   ['layer3_qkv_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer3_qkv_bias_add
#   Op Type:   Add
#   Inputs:    ['layer3_qkv_matmul.output', 'layer3_qkv_bias']
#   Outputs:   ['layer3_qkv_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer3_qkv_split
#   Op Type:   Split
#   Inputs:    ['layer3_qkv_bias_add.output']
#   Outputs:   ['layer3_q.output', 'layer3_k.output', 'layer3_v.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer3_q_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer3_q.output', 'layer3_q_shape']
#   Outputs:   ['layer3_q_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer3_q_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer3_q_reshape.output']
#   Outputs:   ['layer3_q_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer3_k_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer3_k.output', 'layer3_k_shape']
#   Outputs:   ['layer3_k_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer3_k_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer3_k_reshape.output']
#   Outputs:   ['layer3_k_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer3_v_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer3_v.output', 'layer3_v_shape']
#   Outputs:   ['layer3_v_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer3_v_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer3_v_reshape.output']
#   Outputs:   ['layer3_v_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer3_attn_matmul_qk
#   Op Type:   MatMul
#   Inputs:    ['layer3_q_transpose.output', 'layer3_k_transpose.output']
#   Outputs:   ['layer3_attn_matmul_qk.output']
#   Attributes: []
# Node info:
#   Name:      layer3_attn_scale
#   Op Type:   Mul
#   Inputs:    ['layer3_attn_matmul_qk.output', 'layer3_scale']
#   Outputs:   ['layer3_attn_scaled.output']
#   Attributes: []
# Node info:
#   Name:      layer3_attn_softmax
#   Op Type:   Softmax
#   Inputs:    ['layer3_attn_scaled.output']
#   Outputs:   ['layer3_attn_softmax.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer3_attn_matmul_v
#   Op Type:   MatMul
#   Inputs:    ['layer3_attn_softmax.output', 'layer3_v_transpose.output']
#   Outputs:   ['layer3_attn_matmul_v.output']
#   Attributes: []
# Node info:
#   Name:      layer3_attn_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer3_attn_matmul_v.output']
#   Outputs:   ['layer3_attn_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer3_attn_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer3_attn_transpose.output', 'layer3_attn_reshape_shape']
#   Outputs:   ['layer3_attn_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer3_proj_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer3_attn_reshape.output', 'layer3_proj_weight']
#   Outputs:   ['layer3_proj_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer3_proj_bias_add
#   Op Type:   Add
#   Inputs:    ['layer3_proj_matmul.output', 'layer3_proj_bias']
#   Outputs:   ['layer3_proj_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer3_attn_add
#   Op Type:   Add
#   Inputs:    ['layer2_ff_add.output', 'layer3_proj_bias_add.output']
#   Outputs:   ['layer3_attn_add.output']
#   Attributes: []
# Node info:
#   Name:      layer3_norm2
#   Op Type:   LayerNormalization
#   Inputs:    ['layer3_attn_add.output', 'layer3_ln2_scale', 'layer3_ln2_bias']
#   Outputs:   ['layer3_norm2.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer3_ff_matmul1
#   Op Type:   MatMul
#   Inputs:    ['layer3_norm2.output', 'layer3_ff_weight1']
#   Outputs:   ['layer3_ff_matmul1.output']
#   Attributes: []
# Node info:
#   Name:      layer3_ff_bias1_add
#   Op Type:   Add
#   Inputs:    ['layer3_ff_matmul1.output', 'layer3_ff_bias1']
#   Outputs:   ['layer3_ff_bias1_add.output']
#   Attributes: []
# Node info:
#   Name:      layer3_ff_relu
#   Op Type:   Relu
#   Inputs:    ['layer3_ff_bias1_add.output']
#   Outputs:   ['layer3_ff_relu.output']
#   Attributes: []
# Node info:
#   Name:      layer3_ff_matmul2
#   Op Type:   MatMul
#   Inputs:    ['layer3_ff_relu.output', 'layer3_ff_weight2']
#   Outputs:   ['layer3_ff_matmul2.output']
#   Attributes: []
# Node info:
#   Name:      layer3_ff_bias2_add
#   Op Type:   Add
#   Inputs:    ['layer3_ff_matmul2.output', 'layer3_ff_bias2']
#   Outputs:   ['layer3_ff_bias2_add.output']
#   Attributes: []
# Node info:
#   Name:      layer3_ff_add
#   Op Type:   Add
#   Inputs:    ['layer3_attn_add.output', 'layer3_ff_bias2_add.output']
#   Outputs:   ['layer3_ff_add.output']
#   Attributes: []
# Node info:
#   Name:      layer4_ln1
#   Op Type:   LayerNormalization
#   Inputs:    ['layer3_ff_add.output', 'layer4_ln1_scale', 'layer4_ln1_bias']
#   Outputs:   ['layer4_norm1.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer4_qkv_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer4_norm1.output', 'layer4_qkv_weight']
#   Outputs:   ['layer4_qkv_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer4_qkv_bias_add
#   Op Type:   Add
#   Inputs:    ['layer4_qkv_matmul.output', 'layer4_qkv_bias']
#   Outputs:   ['layer4_qkv_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer4_qkv_split
#   Op Type:   Split
#   Inputs:    ['layer4_qkv_bias_add.output']
#   Outputs:   ['layer4_q.output', 'layer4_k.output', 'layer4_v.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer4_q_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer4_q.output', 'layer4_q_shape']
#   Outputs:   ['layer4_q_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer4_q_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer4_q_reshape.output']
#   Outputs:   ['layer4_q_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer4_k_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer4_k.output', 'layer4_k_shape']
#   Outputs:   ['layer4_k_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer4_k_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer4_k_reshape.output']
#   Outputs:   ['layer4_k_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer4_v_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer4_v.output', 'layer4_v_shape']
#   Outputs:   ['layer4_v_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer4_v_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer4_v_reshape.output']
#   Outputs:   ['layer4_v_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer4_attn_matmul_qk
#   Op Type:   MatMul
#   Inputs:    ['layer4_q_transpose.output', 'layer4_k_transpose.output']
#   Outputs:   ['layer4_attn_matmul_qk.output']
#   Attributes: []
# Node info:
#   Name:      layer4_attn_scale
#   Op Type:   Mul
#   Inputs:    ['layer4_attn_matmul_qk.output', 'layer4_scale']
#   Outputs:   ['layer4_attn_scaled.output']
#   Attributes: []
# Node info:
#   Name:      layer4_attn_softmax
#   Op Type:   Softmax
#   Inputs:    ['layer4_attn_scaled.output']
#   Outputs:   ['layer4_attn_softmax.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer4_attn_matmul_v
#   Op Type:   MatMul
#   Inputs:    ['layer4_attn_softmax.output', 'layer4_v_transpose.output']
#   Outputs:   ['layer4_attn_matmul_v.output']
#   Attributes: []
# Node info:
#   Name:      layer4_attn_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer4_attn_matmul_v.output']
#   Outputs:   ['layer4_attn_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer4_attn_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer4_attn_transpose.output', 'layer4_attn_reshape_shape']
#   Outputs:   ['layer4_attn_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer4_proj_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer4_attn_reshape.output', 'layer4_proj_weight']
#   Outputs:   ['layer4_proj_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer4_proj_bias_add
#   Op Type:   Add
#   Inputs:    ['layer4_proj_matmul.output', 'layer4_proj_bias']
#   Outputs:   ['layer4_proj_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer4_attn_add
#   Op Type:   Add
#   Inputs:    ['layer3_ff_add.output', 'layer4_proj_bias_add.output']
#   Outputs:   ['layer4_attn_add.output']
#   Attributes: []
# Node info:
#   Name:      layer4_norm2
#   Op Type:   LayerNormalization
#   Inputs:    ['layer4_attn_add.output', 'layer4_ln2_scale', 'layer4_ln2_bias']
#   Outputs:   ['layer4_norm2.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer4_ff_matmul1
#   Op Type:   MatMul
#   Inputs:    ['layer4_norm2.output', 'layer4_ff_weight1']
#   Outputs:   ['layer4_ff_matmul1.output']
#   Attributes: []
# Node info:
#   Name:      layer4_ff_bias1_add
#   Op Type:   Add
#   Inputs:    ['layer4_ff_matmul1.output', 'layer4_ff_bias1']
#   Outputs:   ['layer4_ff_bias1_add.output']
#   Attributes: []
# Node info:
#   Name:      layer4_ff_relu
#   Op Type:   Relu
#   Inputs:    ['layer4_ff_bias1_add.output']
#   Outputs:   ['layer4_ff_relu.output']
#   Attributes: []
# Node info:
#   Name:      layer4_ff_matmul2
#   Op Type:   MatMul
#   Inputs:    ['layer4_ff_relu.output', 'layer4_ff_weight2']
#   Outputs:   ['layer4_ff_matmul2.output']
#   Attributes: []
# Node info:
#   Name:      layer4_ff_bias2_add
#   Op Type:   Add
#   Inputs:    ['layer4_ff_matmul2.output', 'layer4_ff_bias2']
#   Outputs:   ['layer4_ff_bias2_add.output']
#   Attributes: []
# Node info:
#   Name:      layer4_ff_add
#   Op Type:   Add
#   Inputs:    ['layer4_attn_add.output', 'layer4_ff_bias2_add.output']
#   Outputs:   ['layer4_ff_add.output']
#   Attributes: []
# Node info:
#   Name:      layer5_ln1
#   Op Type:   LayerNormalization
#   Inputs:    ['layer4_ff_add.output', 'layer5_ln1_scale', 'layer5_ln1_bias']
#   Outputs:   ['layer5_norm1.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer5_qkv_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer5_norm1.output', 'layer5_qkv_weight']
#   Outputs:   ['layer5_qkv_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer5_qkv_bias_add
#   Op Type:   Add
#   Inputs:    ['layer5_qkv_matmul.output', 'layer5_qkv_bias']
#   Outputs:   ['layer5_qkv_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer5_qkv_split
#   Op Type:   Split
#   Inputs:    ['layer5_qkv_bias_add.output']
#   Outputs:   ['layer5_q.output', 'layer5_k.output', 'layer5_v.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer5_q_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer5_q.output', 'layer5_q_shape']
#   Outputs:   ['layer5_q_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer5_q_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer5_q_reshape.output']
#   Outputs:   ['layer5_q_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer5_k_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer5_k.output', 'layer5_k_shape']
#   Outputs:   ['layer5_k_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer5_k_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer5_k_reshape.output']
#   Outputs:   ['layer5_k_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer5_v_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer5_v.output', 'layer5_v_shape']
#   Outputs:   ['layer5_v_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer5_v_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer5_v_reshape.output']
#   Outputs:   ['layer5_v_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer5_attn_matmul_qk
#   Op Type:   MatMul
#   Inputs:    ['layer5_q_transpose.output', 'layer5_k_transpose.output']
#   Outputs:   ['layer5_attn_matmul_qk.output']
#   Attributes: []
# Node info:
#   Name:      layer5_attn_scale
#   Op Type:   Mul
#   Inputs:    ['layer5_attn_matmul_qk.output', 'layer5_scale']
#   Outputs:   ['layer5_attn_scaled.output']
#   Attributes: []
# Node info:
#   Name:      layer5_attn_softmax
#   Op Type:   Softmax
#   Inputs:    ['layer5_attn_scaled.output']
#   Outputs:   ['layer5_attn_softmax.output']
#   Attributes: ['axis: 0.0']
# Node info:
#   Name:      layer5_attn_matmul_v
#   Op Type:   MatMul
#   Inputs:    ['layer5_attn_softmax.output', 'layer5_v_transpose.output']
#   Outputs:   ['layer5_attn_matmul_v.output']
#   Attributes: []
# Node info:
#   Name:      layer5_attn_transpose
#   Op Type:   Transpose
#   Inputs:    ['layer5_attn_matmul_v.output']
#   Outputs:   ['layer5_attn_transpose.output']
#   Attributes: ['perm: [0, 2, 1, 3]']
# Node info:
#   Name:      layer5_attn_reshape
#   Op Type:   Reshape
#   Inputs:    ['layer5_attn_transpose.output', 'layer5_attn_reshape_shape']
#   Outputs:   ['layer5_attn_reshape.output']
#   Attributes: []
# Node info:
#   Name:      layer5_proj_matmul
#   Op Type:   MatMul
#   Inputs:    ['layer5_attn_reshape.output', 'layer5_proj_weight']
#   Outputs:   ['layer5_proj_matmul.output']
#   Attributes: []
# Node info:
#   Name:      layer5_proj_bias_add
#   Op Type:   Add
#   Inputs:    ['layer5_proj_matmul.output', 'layer5_proj_bias']
#   Outputs:   ['layer5_proj_bias_add.output']
#   Attributes: []
# Node info:
#   Name:      layer5_attn_add
#   Op Type:   Add
#   Inputs:    ['layer4_ff_add.output', 'layer5_proj_bias_add.output']
#   Outputs:   ['layer5_attn_add.output']
#   Attributes: []
# Node info:
#   Name:      layer5_norm2
#   Op Type:   LayerNormalization
#   Inputs:    ['layer5_attn_add.output', 'layer5_ln2_scale', 'layer5_ln2_bias']
#   Outputs:   ['layer5_norm2.output']
#   Attributes: ['epsilon: 9.999999747378752e-06']
# Node info:
#   Name:      layer5_ff_matmul1
#   Op Type:   MatMul
#   Inputs:    ['layer5_norm2.output', 'layer5_ff_weight1']
#   Outputs:   ['layer5_ff_matmul1.output']
#   Attributes: []
# Node info:
#   Name:      layer5_ff_bias1_add
#   Op Type:   Add
#   Inputs:    ['layer5_ff_matmul1.output', 'layer5_ff_bias1']
#   Outputs:   ['layer5_ff_bias1_add.output']
#   Attributes: []
# Node info:
#   Name:      layer5_ff_relu
#   Op Type:   Relu
#   Inputs:    ['layer5_ff_bias1_add.output']
#   Outputs:   ['layer5_ff_relu.output']
#   Attributes: []
# Node info:
#   Name:      layer5_ff_matmul2
#   Op Type:   MatMul
#   Inputs:    ['layer5_ff_relu.output', 'layer5_ff_weight2']
#   Outputs:   ['layer5_ff_matmul2.output']
#   Attributes: []
# Node info:
#   Name:      layer5_ff_bias2_add
#   Op Type:   Add
#   Inputs:    ['layer5_ff_matmul2.output', 'layer5_ff_bias2']
#   Outputs:   ['layer5_ff_bias2_add.output']
#   Attributes: []
# Node info:
#   Name:      layer5_ff_add
#   Op Type:   Add
#   Inputs:    ['layer5_attn_add.output', 'layer5_ff_bias2_add.output']
#   Outputs:   ['layer5_ff_add.output']
#   Attributes: []
# Node info:
#   Name:      ln_final
#   Op Type:   LayerNormalization
#   Inputs:    ['layer5_ff_add.output', 'ln_final_scale', 'ln_final_bias']
#   Outputs:   ['output0']
#   Attributes: ['epsilon: 9.999999747378752e-06']

# ==================================================
# Parsing Initializer Information
# ==================================================
# Initializer info:
#   Name:      layer0_ln1_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer0_ln1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer0_qkv_weight
#   Data Type: FLOAT
#   Shape:     [1536, 512]
# Initializer info:
#   Name:      layer0_qkv_bias
#   Data Type: FLOAT
#   Shape:     [1536]
# Initializer info:
#   Name:      layer0_proj_weight
#   Data Type: FLOAT
#   Shape:     [512, 512]
# Initializer info:
#   Name:      layer0_proj_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer0_q_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer0_k_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer0_v_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer0_scale
#   Data Type: FLOAT
#   Shape:     [1]
# Initializer info:
#   Name:      layer0_attn_reshape_shape
#   Data Type: INT64
#   Shape:     [3]
# Initializer info:
#   Name:      layer0_ln2_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer0_ln2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer0_ff_weight1
#   Data Type: FLOAT
#   Shape:     [2048, 512]
# Initializer info:
#   Name:      layer0_ff_bias1
#   Data Type: FLOAT
#   Shape:     [2048]
# Initializer info:
#   Name:      layer0_ff_weight2
#   Data Type: FLOAT
#   Shape:     [512, 2048]
# Initializer info:
#   Name:      layer0_ff_bias2
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer1_ln1_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer1_ln1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer1_qkv_weight
#   Data Type: FLOAT
#   Shape:     [1536, 512]
# Initializer info:
#   Name:      layer1_qkv_bias
#   Data Type: FLOAT
#   Shape:     [1536]
# Initializer info:
#   Name:      layer1_proj_weight
#   Data Type: FLOAT
#   Shape:     [512, 512]
# Initializer info:
#   Name:      layer1_proj_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer1_q_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer1_k_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer1_v_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer1_scale
#   Data Type: FLOAT
#   Shape:     [1]
# Initializer info:
#   Name:      layer1_attn_reshape_shape
#   Data Type: INT64
#   Shape:     [3]
# Initializer info:
#   Name:      layer1_ln2_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer1_ln2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer1_ff_weight1
#   Data Type: FLOAT
#   Shape:     [2048, 512]
# Initializer info:
#   Name:      layer1_ff_bias1
#   Data Type: FLOAT
#   Shape:     [2048]
# Initializer info:
#   Name:      layer1_ff_weight2
#   Data Type: FLOAT
#   Shape:     [512, 2048]
# Initializer info:
#   Name:      layer1_ff_bias2
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer2_ln1_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer2_ln1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer2_qkv_weight
#   Data Type: FLOAT
#   Shape:     [1536, 512]
# Initializer info:
#   Name:      layer2_qkv_bias
#   Data Type: FLOAT
#   Shape:     [1536]
# Initializer info:
#   Name:      layer2_proj_weight
#   Data Type: FLOAT
#   Shape:     [512, 512]
# Initializer info:
#   Name:      layer2_proj_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer2_q_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer2_k_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer2_v_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer2_scale
#   Data Type: FLOAT
#   Shape:     [1]
# Initializer info:
#   Name:      layer2_attn_reshape_shape
#   Data Type: INT64
#   Shape:     [3]
# Initializer info:
#   Name:      layer2_ln2_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer2_ln2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer2_ff_weight1
#   Data Type: FLOAT
#   Shape:     [2048, 512]
# Initializer info:
#   Name:      layer2_ff_bias1
#   Data Type: FLOAT
#   Shape:     [2048]
# Initializer info:
#   Name:      layer2_ff_weight2
#   Data Type: FLOAT
#   Shape:     [512, 2048]
# Initializer info:
#   Name:      layer2_ff_bias2
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer3_ln1_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer3_ln1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer3_qkv_weight
#   Data Type: FLOAT
#   Shape:     [1536, 512]
# Initializer info:
#   Name:      layer3_qkv_bias
#   Data Type: FLOAT
#   Shape:     [1536]
# Initializer info:
#   Name:      layer3_proj_weight
#   Data Type: FLOAT
#   Shape:     [512, 512]
# Initializer info:
#   Name:      layer3_proj_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer3_q_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer3_k_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer3_v_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer3_scale
#   Data Type: FLOAT
#   Shape:     [1]
# Initializer info:
#   Name:      layer3_attn_reshape_shape
#   Data Type: INT64
#   Shape:     [3]
# Initializer info:
#   Name:      layer3_ln2_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer3_ln2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer3_ff_weight1
#   Data Type: FLOAT
#   Shape:     [2048, 512]
# Initializer info:
#   Name:      layer3_ff_bias1
#   Data Type: FLOAT
#   Shape:     [2048]
# Initializer info:
#   Name:      layer3_ff_weight2
#   Data Type: FLOAT
#   Shape:     [512, 2048]
# Initializer info:
#   Name:      layer3_ff_bias2
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer4_ln1_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer4_ln1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer4_qkv_weight
#   Data Type: FLOAT
#   Shape:     [1536, 512]
# Initializer info:
#   Name:      layer4_qkv_bias
#   Data Type: FLOAT
#   Shape:     [1536]
# Initializer info:
#   Name:      layer4_proj_weight
#   Data Type: FLOAT
#   Shape:     [512, 512]
# Initializer info:
#   Name:      layer4_proj_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer4_q_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer4_k_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer4_v_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer4_scale
#   Data Type: FLOAT
#   Shape:     [1]
# Initializer info:
#   Name:      layer4_attn_reshape_shape
#   Data Type: INT64
#   Shape:     [3]
# Initializer info:
#   Name:      layer4_ln2_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer4_ln2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer4_ff_weight1
#   Data Type: FLOAT
#   Shape:     [2048, 512]
# Initializer info:
#   Name:      layer4_ff_bias1
#   Data Type: FLOAT
#   Shape:     [2048]
# Initializer info:
#   Name:      layer4_ff_weight2
#   Data Type: FLOAT
#   Shape:     [512, 2048]
# Initializer info:
#   Name:      layer4_ff_bias2
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer5_ln1_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer5_ln1_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer5_qkv_weight
#   Data Type: FLOAT
#   Shape:     [1536, 512]
# Initializer info:
#   Name:      layer5_qkv_bias
#   Data Type: FLOAT
#   Shape:     [1536]
# Initializer info:
#   Name:      layer5_proj_weight
#   Data Type: FLOAT
#   Shape:     [512, 512]
# Initializer info:
#   Name:      layer5_proj_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer5_q_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer5_k_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer5_v_shape
#   Data Type: INT64
#   Shape:     [4]
# Initializer info:
#   Name:      layer5_scale
#   Data Type: FLOAT
#   Shape:     [1]
# Initializer info:
#   Name:      layer5_attn_reshape_shape
#   Data Type: INT64
#   Shape:     [3]
# Initializer info:
#   Name:      layer5_ln2_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer5_ln2_bias
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      layer5_ff_weight1
#   Data Type: FLOAT
#   Shape:     [2048, 512]
# Initializer info:
#   Name:      layer5_ff_bias1
#   Data Type: FLOAT
#   Shape:     [2048]
# Initializer info:
#   Name:      layer5_ff_weight2
#   Data Type: FLOAT
#   Shape:     [512, 2048]
# Initializer info:
#   Name:      layer5_ff_bias2
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      ln_final_scale
#   Data Type: FLOAT
#   Shape:     [512]
# Initializer info:
#   Name:      ln_final_bias
#   Data Type: FLOAT
#   Shape:     [512]

# Total nodes: 157
# Total initializers: 104
