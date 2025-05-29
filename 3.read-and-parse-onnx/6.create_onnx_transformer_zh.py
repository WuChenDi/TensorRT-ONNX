import numpy as np
import onnx
from onnx import helper, TensorProto
from utils import get_onnx_path

def create_initializer_tensor(
    name: str,
    tensor_array: np.ndarray,
    data_type: int = TensorProto.FLOAT
) -> onnx.TensorProto:
    """创建ONNX TensorProto初始化张量。

    参数:
        name: 初始化器的名称。
        tensor_array: 包含张量数据的Numpy数组。
        data_type: ONNX数据类型（例如，TensorProto.FLOAT）。

    返回:
        TensorProto: ONNX初始化张量。
    """
    initializer = helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist()
    )
    return initializer

def parse_onnx(model: onnx.ModelProto) -> None:
    """解析并打印ONNX模型结构。

    参数:
        model: 要解析的ONNX模型。

    注意:
        - 从model.graph中解析输入、输出、节点和初始化权重。
        - 打印名称、数据类型、形状和节点属性。
    """
    try:
        graph = model.graph
        if not graph:
            raise ValueError("模型计算图为空")

        print(f"\n{'='*50}")
        print("解析输入信息")
        print(f"{'='*50}")
        for input in graph.input:
            input_shape = [d.dim_value if d.dim_value != 0 else None for d in input.type.tensor_type.shape.dim]
            print(f"输入信息:\n"
                  f"  名称:     {input.name}\n"
                  f"  数据类型: {TensorProto.DataType.Name(input.type.tensor_type.elem_type)}\n"
                  f"  形状:     {input_shape}")

        print(f"\n{'='*50}")
        print("解析输出信息")
        print(f"{'='*50}")
        for output in graph.output:
            output_shape = [d.dim_value if d.dim_value != 0 else None for d in output.type.tensor_type.shape.dim]
            print(f"输出信息:\n"
                  f"  名称:     {output.name}\n"
                  f"  数据类型: {TensorProto.DataType.Name(output.type.tensor_type.elem_type)}\n"
                  f"  形状:     {output_shape}")

        print(f"\n{'='*50}")
        print("解析节点信息")
        print(f"{'='*50}")
        for node in graph.node:
            attributes = []
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INTS:
                    value = attr.ints
                elif attr.type == onnx.AttributeProto.FLOATS:
                    value = attr.floats
                elif attr.type == onnx.AttributeProto.FLOAT:
                    value = attr.f
                elif attr.type == onnx.AttributeProto.STRING:
                    value = attr.s.decode()
                else:
                    value = "不支持"
                attributes.append(f"{attr.name}: {value}")
            print(f"节点信息:\n"
                  f"  名称:     {node.name}\n"
                  f"  操作类型: {node.op_type}\n"
                  f"  输入:     {node.input}\n"
                  f"  输出:     {node.output}\n"
                  f"  属性:     {attributes}")

        print(f"\n{'='*50}")
        print("解析初始化权重信息")
        print(f"{'='*50}")
        for initializer in graph.initializer:
            print(f"初始化权重信息:\n"
                  f"  名称:     {initializer.name}\n"
                  f"  数据类型: {TensorProto.DataType.Name(initializer.data_type)}\n"
                  f"  形状:     {list(initializer.dims)}")
    except Exception as e:
        print(f"解析模型失败: {str(e)}")

def main():
    """创建并保存Transformer ONNX模型。"""
    # 模型配置
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

    # 创建输入和输出
    model_input_name = "input0"
    model_output_name = "output0"

    input = helper.make_tensor_value_info(
        model_input_name, TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(
        model_output_name, TensorProto.FLOAT, output_shape)

    nodes = []
    initializers = []
    previous_output_name = model_input_name

    # 创建Transformer层
    for layer in range(num_layers):
        layer_norm1_output = f"layer{layer}_norm1.output"
        attn_output = f"layer{layer}_attn.output"
        attn_output_add = f"layer{layer}_attn_add.output"
        layer_norm2_output = f"layer{layer}_norm2.output"
        ff_output = f"layer{layer}_ff.output"
        ff_output_add = f"layer{layer}_ff_add.output"

        # 层归一化 1
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
            name=f"layer{layer}_norm1",
            epsilon=1e-6
        )
        nodes.append(ln1_node)

        # 多头注意力
        qkv_weight = np.random.randn(input_dim, 3 * input_dim).astype(np.float32) * np.sqrt(2.0 / (input_dim + 3 * input_dim))
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

        # QKV 运算
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

        # 分割 Q, K, V
        q_output = f"layer{layer}_q.output"
        k_output = f"layer{layer}_k.output"
        v_output = f"layer{layer}_v.output"

        split_outputs = [q_output, k_output, v_output]
        split_node = helper.make_node(
            "Split",
            inputs=[qkv_bias_add_output],
            outputs=split_outputs,
            name=f"layer{layer}_qkv_split",
            axis=-1
        )
        nodes.append(split_node)

        # 重塑 Q, K, V 以实现多头注意力
        def reshape_for_heads(name: str, input_name: str):
            reshape_output = f"{name}_reshape.output"
            transpose_output = f"{name}_transpose.output"
            shape = np.array([batch_size, seq_length, num_heads, head_dim], dtype=np.int64)
            shape_initializer = create_initializer_tensor(
                f"{name}_shape", shape, TensorProto.INT64)
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

        # 计算注意力分数
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
            axis=-1
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

        # 输出投影
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

        # 残差连接 1
        attn_output_add_node = helper.make_node(
            "Add",
            inputs=[previous_output_name, proj_bias_add_output],
            outputs=[attn_output_add],
            name=f"layer{layer}_attn_add"
        )
        nodes.append(attn_output_add_node)

        # 层归一化 2
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
            epsilon=1e-6
        )
        nodes.append(ln2_node)

        # 前馈网络
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

        # 残差连接 2
        ff_output_add_node = helper.make_node(
            "Add",
            inputs=[attn_output_add, ff_bias2_add_output],
            outputs=[ff_output_add],
            name=f"layer{layer}_ff_add"
        )
        nodes.append(ff_output_add_node)

        previous_output_name = ff_output_add

    # 最终层归一化
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
        epsilon=1e-6
    )
    nodes.append(ln_final_node)

    # 创建计算图
    graph = helper.make_graph(
        nodes=nodes,
        name="transformer",
        inputs=[input],
        outputs=[output],
        initializer=initializers
    )

    # 创建模型
    model = helper.make_model(graph, producer_name="onnx-transformer-sample")
    model.opset_import[0].version = 17

    # 验证并保存模型
    try:
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        output_path = get_onnx_path(__file__, "transformer_zh.onnx")
        onnx.save(model, output_path)
        print(f"成功创建模型: {output_path}")

        # 解析模型以便调试
        parse_onnx(model)
        print(f"\n总节点数: {len(nodes)}")
        print(f"总初始化器数: {len(initializers)}")
    except Exception as e:
        print(f"创建或保存模型失败: {str(e)}")

if __name__ == "__main__":
    main()

# python3 3.read-and-parse-onnx/6.create_onnx_transformer_zh.py
# 成功创建模型: /home/wudi/work/github/WuChenDi/TensorRT-ONNX/3.read-and-parse-onnx/models/transformer_zh.onnx

# ==================================================
# 解析输入信息
# ==================================================
# 输入信息:
#   名称:     input0
#   数据类型: FLOAT
#   形状:     [1, 128, 512]

# ==================================================
# 解析输出信息
# ==================================================
# 输出信息:
#   名称:     output0
#   数据类型: FLOAT
#   形状:     [1, 128, 512]

# ==================================================
# 解析节点信息
# ==================================================
# 节点信息:
#   名称:     layer0_norm1
#   操作类型: LayerNormalization
#   输入:     ['input0', 'layer0_ln1_scale', 'layer0_ln1_bias']
#   输出:     ['layer0_norm1.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer0_qkv_matmul
#   操作类型: MatMul
#   输入:     ['layer0_norm1.output', 'layer0_qkv_weight']
#   输出:     ['layer0_qkv_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_qkv_bias_add
#   操作类型: Add
#   输入:     ['layer0_qkv_matmul.output', 'layer0_qkv_bias']
#   输出:     ['layer0_qkv_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_qkv_split
#   操作类型: Split
#   输入:     ['layer0_qkv_bias_add.output']
#   输出:     ['layer0_q.output', 'layer0_k.output', 'layer0_v.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer0_q_reshape
#   操作类型: Reshape
#   输入:     ['layer0_q.output', 'layer0_q_shape']
#   输出:     ['layer0_q_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_q_transpose
#   操作类型: Transpose
#   输入:     ['layer0_q_reshape.output']
#   输出:     ['layer0_q_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer0_k_reshape
#   操作类型: Reshape
#   输入:     ['layer0_k.output', 'layer0_k_shape']
#   输出:     ['layer0_k_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_k_transpose
#   操作类型: Transpose
#   输入:     ['layer0_k_reshape.output']
#   输出:     ['layer0_k_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer0_v_reshape
#   操作类型: Reshape
#   输入:     ['layer0_v.output', 'layer0_v_shape']
#   输出:     ['layer0_v_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_v_transpose
#   操作类型: Transpose
#   输入:     ['layer0_v_reshape.output']
#   输出:     ['layer0_v_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer0_attn_matmul_qk
#   操作类型: MatMul
#   输入:     ['layer0_q_transpose.output', 'layer0_k_transpose.output']
#   输出:     ['layer0_attn_matmul_qk.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_attn_scale
#   操作类型: Mul
#   输入:     ['layer0_attn_matmul_qk.output', 'layer0_scale']
#   输出:     ['layer0_attn_scaled.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_attn_softmax
#   操作类型: Softmax
#   输入:     ['layer0_attn_scaled.output']
#   输出:     ['layer0_attn_softmax.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer0_attn_matmul_v
#   操作类型: MatMul
#   输入:     ['layer0_attn_softmax.output', 'layer0_v_transpose.output']
#   输出:     ['layer0_attn_matmul_v.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_attn_transpose
#   操作类型: Transpose
#   输入:     ['layer0_attn_matmul_v.output']
#   输出:     ['layer0_attn_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer0_attn_reshape
#   操作类型: Reshape
#   输入:     ['layer0_attn_transpose.output', 'layer0_attn_reshape_shape']
#   输出:     ['layer0_attn_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_proj_matmul
#   操作类型: MatMul
#   输入:     ['layer0_attn_reshape.output', 'layer0_proj_weight']
#   输出:     ['layer0_proj_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_proj_bias_add
#   操作类型: Add
#   输入:     ['layer0_proj_matmul.output', 'layer0_proj_bias']
#   输出:     ['layer0_proj_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_attn_add
#   操作类型: Add
#   输入:     ['input0', 'layer0_proj_bias_add.output']
#   输出:     ['layer0_attn_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_norm2
#   操作类型: LayerNormalization
#   输入:     ['layer0_attn_add.output', 'layer0_ln2_scale', 'layer0_ln2_bias']
#   输出:     ['layer0_norm2.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer0_ff_matmul1
#   操作类型: MatMul
#   输入:     ['layer0_norm2.output', 'layer0_ff_weight1']
#   输出:     ['layer0_ff_matmul1.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_ff_bias1_add
#   操作类型: Add
#   输入:     ['layer0_ff_matmul1.output', 'layer0_ff_bias1']
#   输出:     ['layer0_ff_bias1_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_ff_relu
#   操作类型: Relu
#   输入:     ['layer0_ff_bias1_add.output']
#   输出:     ['layer0_ff_relu.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_ff_matmul2
#   操作类型: MatMul
#   输入:     ['layer0_ff_relu.output', 'layer0_ff_weight2']
#   输出:     ['layer0_ff_matmul2.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_ff_bias2_add
#   操作类型: Add
#   输入:     ['layer0_ff_matmul2.output', 'layer0_ff_bias2']
#   输出:     ['layer0_ff_bias2_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer0_ff_add
#   操作类型: Add
#   输入:     ['layer0_attn_add.output', 'layer0_ff_bias2_add.output']
#   输出:     ['layer0_ff_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_norm1
#   操作类型: LayerNormalization
#   输入:     ['layer0_ff_add.output', 'layer1_ln1_scale', 'layer1_ln1_bias']
#   输出:     ['layer1_norm1.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer1_qkv_matmul
#   操作类型: MatMul
#   输入:     ['layer1_norm1.output', 'layer1_qkv_weight']
#   输出:     ['layer1_qkv_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_qkv_bias_add
#   操作类型: Add
#   输入:     ['layer1_qkv_matmul.output', 'layer1_qkv_bias']
#   输出:     ['layer1_qkv_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_qkv_split
#   操作类型: Split
#   输入:     ['layer1_qkv_bias_add.output']
#   输出:     ['layer1_q.output', 'layer1_k.output', 'layer1_v.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer1_q_reshape
#   操作类型: Reshape
#   输入:     ['layer1_q.output', 'layer1_q_shape']
#   输出:     ['layer1_q_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_q_transpose
#   操作类型: Transpose
#   输入:     ['layer1_q_reshape.output']
#   输出:     ['layer1_q_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer1_k_reshape
#   操作类型: Reshape
#   输入:     ['layer1_k.output', 'layer1_k_shape']
#   输出:     ['layer1_k_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_k_transpose
#   操作类型: Transpose
#   输入:     ['layer1_k_reshape.output']
#   输出:     ['layer1_k_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer1_v_reshape
#   操作类型: Reshape
#   输入:     ['layer1_v.output', 'layer1_v_shape']
#   输出:     ['layer1_v_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_v_transpose
#   操作类型: Transpose
#   输入:     ['layer1_v_reshape.output']
#   输出:     ['layer1_v_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer1_attn_matmul_qk
#   操作类型: MatMul
#   输入:     ['layer1_q_transpose.output', 'layer1_k_transpose.output']
#   输出:     ['layer1_attn_matmul_qk.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_attn_scale
#   操作类型: Mul
#   输入:     ['layer1_attn_matmul_qk.output', 'layer1_scale']
#   输出:     ['layer1_attn_scaled.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_attn_softmax
#   操作类型: Softmax
#   输入:     ['layer1_attn_scaled.output']
#   输出:     ['layer1_attn_softmax.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer1_attn_matmul_v
#   操作类型: MatMul
#   输入:     ['layer1_attn_softmax.output', 'layer1_v_transpose.output']
#   输出:     ['layer1_attn_matmul_v.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_attn_transpose
#   操作类型: Transpose
#   输入:     ['layer1_attn_matmul_v.output']
#   输出:     ['layer1_attn_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer1_attn_reshape
#   操作类型: Reshape
#   输入:     ['layer1_attn_transpose.output', 'layer1_attn_reshape_shape']
#   输出:     ['layer1_attn_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_proj_matmul
#   操作类型: MatMul
#   输入:     ['layer1_attn_reshape.output', 'layer1_proj_weight']
#   输出:     ['layer1_proj_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_proj_bias_add
#   操作类型: Add
#   输入:     ['layer1_proj_matmul.output', 'layer1_proj_bias']
#   输出:     ['layer1_proj_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_attn_add
#   操作类型: Add
#   输入:     ['layer0_ff_add.output', 'layer1_proj_bias_add.output']
#   输出:     ['layer1_attn_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_norm2
#   操作类型: LayerNormalization
#   输入:     ['layer1_attn_add.output', 'layer1_ln2_scale', 'layer1_ln2_bias']
#   输出:     ['layer1_norm2.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer1_ff_matmul1
#   操作类型: MatMul
#   输入:     ['layer1_norm2.output', 'layer1_ff_weight1']
#   输出:     ['layer1_ff_matmul1.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_ff_bias1_add
#   操作类型: Add
#   输入:     ['layer1_ff_matmul1.output', 'layer1_ff_bias1']
#   输出:     ['layer1_ff_bias1_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_ff_relu
#   操作类型: Relu
#   输入:     ['layer1_ff_bias1_add.output']
#   输出:     ['layer1_ff_relu.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_ff_matmul2
#   操作类型: MatMul
#   输入:     ['layer1_ff_relu.output', 'layer1_ff_weight2']
#   输出:     ['layer1_ff_matmul2.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_ff_bias2_add
#   操作类型: Add
#   输入:     ['layer1_ff_matmul2.output', 'layer1_ff_bias2']
#   输出:     ['layer1_ff_bias2_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer1_ff_add
#   操作类型: Add
#   输入:     ['layer1_attn_add.output', 'layer1_ff_bias2_add.output']
#   输出:     ['layer1_ff_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_norm1
#   操作类型: LayerNormalization
#   输入:     ['layer1_ff_add.output', 'layer2_ln1_scale', 'layer2_ln1_bias']
#   输出:     ['layer2_norm1.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer2_qkv_matmul
#   操作类型: MatMul
#   输入:     ['layer2_norm1.output', 'layer2_qkv_weight']
#   输出:     ['layer2_qkv_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_qkv_bias_add
#   操作类型: Add
#   输入:     ['layer2_qkv_matmul.output', 'layer2_qkv_bias']
#   输出:     ['layer2_qkv_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_qkv_split
#   操作类型: Split
#   输入:     ['layer2_qkv_bias_add.output']
#   输出:     ['layer2_q.output', 'layer2_k.output', 'layer2_v.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer2_q_reshape
#   操作类型: Reshape
#   输入:     ['layer2_q.output', 'layer2_q_shape']
#   输出:     ['layer2_q_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_q_transpose
#   操作类型: Transpose
#   输入:     ['layer2_q_reshape.output']
#   输出:     ['layer2_q_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer2_k_reshape
#   操作类型: Reshape
#   输入:     ['layer2_k.output', 'layer2_k_shape']
#   输出:     ['layer2_k_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_k_transpose
#   操作类型: Transpose
#   输入:     ['layer2_k_reshape.output']
#   输出:     ['layer2_k_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer2_v_reshape
#   操作类型: Reshape
#   输入:     ['layer2_v.output', 'layer2_v_shape']
#   输出:     ['layer2_v_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_v_transpose
#   操作类型: Transpose
#   输入:     ['layer2_v_reshape.output']
#   输出:     ['layer2_v_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer2_attn_matmul_qk
#   操作类型: MatMul
#   输入:     ['layer2_q_transpose.output', 'layer2_k_transpose.output']
#   输出:     ['layer2_attn_matmul_qk.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_attn_scale
#   操作类型: Mul
#   输入:     ['layer2_attn_matmul_qk.output', 'layer2_scale']
#   输出:     ['layer2_attn_scaled.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_attn_softmax
#   操作类型: Softmax
#   输入:     ['layer2_attn_scaled.output']
#   输出:     ['layer2_attn_softmax.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer2_attn_matmul_v
#   操作类型: MatMul
#   输入:     ['layer2_attn_softmax.output', 'layer2_v_transpose.output']
#   输出:     ['layer2_attn_matmul_v.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_attn_transpose
#   操作类型: Transpose
#   输入:     ['layer2_attn_matmul_v.output']
#   输出:     ['layer2_attn_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer2_attn_reshape
#   操作类型: Reshape
#   输入:     ['layer2_attn_transpose.output', 'layer2_attn_reshape_shape']
#   输出:     ['layer2_attn_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_proj_matmul
#   操作类型: MatMul
#   输入:     ['layer2_attn_reshape.output', 'layer2_proj_weight']
#   输出:     ['layer2_proj_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_proj_bias_add
#   操作类型: Add
#   输入:     ['layer2_proj_matmul.output', 'layer2_proj_bias']
#   输出:     ['layer2_proj_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_attn_add
#   操作类型: Add
#   输入:     ['layer1_ff_add.output', 'layer2_proj_bias_add.output']
#   输出:     ['layer2_attn_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_norm2
#   操作类型: LayerNormalization
#   输入:     ['layer2_attn_add.output', 'layer2_ln2_scale', 'layer2_ln2_bias']
#   输出:     ['layer2_norm2.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer2_ff_matmul1
#   操作类型: MatMul
#   输入:     ['layer2_norm2.output', 'layer2_ff_weight1']
#   输出:     ['layer2_ff_matmul1.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_ff_bias1_add
#   操作类型: Add
#   输入:     ['layer2_ff_matmul1.output', 'layer2_ff_bias1']
#   输出:     ['layer2_ff_bias1_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_ff_relu
#   操作类型: Relu
#   输入:     ['layer2_ff_bias1_add.output']
#   输出:     ['layer2_ff_relu.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_ff_matmul2
#   操作类型: MatMul
#   输入:     ['layer2_ff_relu.output', 'layer2_ff_weight2']
#   输出:     ['layer2_ff_matmul2.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_ff_bias2_add
#   操作类型: Add
#   输入:     ['layer2_ff_matmul2.output', 'layer2_ff_bias2']
#   输出:     ['layer2_ff_bias2_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer2_ff_add
#   操作类型: Add
#   输入:     ['layer2_attn_add.output', 'layer2_ff_bias2_add.output']
#   输出:     ['layer2_ff_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_norm1
#   操作类型: LayerNormalization
#   输入:     ['layer2_ff_add.output', 'layer3_ln1_scale', 'layer3_ln1_bias']
#   输出:     ['layer3_norm1.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer3_qkv_matmul
#   操作类型: MatMul
#   输入:     ['layer3_norm1.output', 'layer3_qkv_weight']
#   输出:     ['layer3_qkv_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_qkv_bias_add
#   操作类型: Add
#   输入:     ['layer3_qkv_matmul.output', 'layer3_qkv_bias']
#   输出:     ['layer3_qkv_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_qkv_split
#   操作类型: Split
#   输入:     ['layer3_qkv_bias_add.output']
#   输出:     ['layer3_q.output', 'layer3_k.output', 'layer3_v.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer3_q_reshape
#   操作类型: Reshape
#   输入:     ['layer3_q.output', 'layer3_q_shape']
#   输出:     ['layer3_q_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_q_transpose
#   操作类型: Transpose
#   输入:     ['layer3_q_reshape.output']
#   输出:     ['layer3_q_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer3_k_reshape
#   操作类型: Reshape
#   输入:     ['layer3_k.output', 'layer3_k_shape']
#   输出:     ['layer3_k_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_k_transpose
#   操作类型: Transpose
#   输入:     ['layer3_k_reshape.output']
#   输出:     ['layer3_k_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer3_v_reshape
#   操作类型: Reshape
#   输入:     ['layer3_v.output', 'layer3_v_shape']
#   输出:     ['layer3_v_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_v_transpose
#   操作类型: Transpose
#   输入:     ['layer3_v_reshape.output']
#   输出:     ['layer3_v_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer3_attn_matmul_qk
#   操作类型: MatMul
#   输入:     ['layer3_q_transpose.output', 'layer3_k_transpose.output']
#   输出:     ['layer3_attn_matmul_qk.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_attn_scale
#   操作类型: Mul
#   输入:     ['layer3_attn_matmul_qk.output', 'layer3_scale']
#   输出:     ['layer3_attn_scaled.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_attn_softmax
#   操作类型: Softmax
#   输入:     ['layer3_attn_scaled.output']
#   输出:     ['layer3_attn_softmax.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer3_attn_matmul_v
#   操作类型: MatMul
#   输入:     ['layer3_attn_softmax.output', 'layer3_v_transpose.output']
#   输出:     ['layer3_attn_matmul_v.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_attn_transpose
#   操作类型: Transpose
#   输入:     ['layer3_attn_matmul_v.output']
#   输出:     ['layer3_attn_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer3_attn_reshape
#   操作类型: Reshape
#   输入:     ['layer3_attn_transpose.output', 'layer3_attn_reshape_shape']
#   输出:     ['layer3_attn_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_proj_matmul
#   操作类型: MatMul
#   输入:     ['layer3_attn_reshape.output', 'layer3_proj_weight']
#   输出:     ['layer3_proj_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_proj_bias_add
#   操作类型: Add
#   输入:     ['layer3_proj_matmul.output', 'layer3_proj_bias']
#   输出:     ['layer3_proj_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_attn_add
#   操作类型: Add
#   输入:     ['layer2_ff_add.output', 'layer3_proj_bias_add.output']
#   输出:     ['layer3_attn_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_norm2
#   操作类型: LayerNormalization
#   输入:     ['layer3_attn_add.output', 'layer3_ln2_scale', 'layer3_ln2_bias']
#   输出:     ['layer3_norm2.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer3_ff_matmul1
#   操作类型: MatMul
#   输入:     ['layer3_norm2.output', 'layer3_ff_weight1']
#   输出:     ['layer3_ff_matmul1.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_ff_bias1_add
#   操作类型: Add
#   输入:     ['layer3_ff_matmul1.output', 'layer3_ff_bias1']
#   输出:     ['layer3_ff_bias1_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_ff_relu
#   操作类型: Relu
#   输入:     ['layer3_ff_bias1_add.output']
#   输出:     ['layer3_ff_relu.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_ff_matmul2
#   操作类型: MatMul
#   输入:     ['layer3_ff_relu.output', 'layer3_ff_weight2']
#   输出:     ['layer3_ff_matmul2.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_ff_bias2_add
#   操作类型: Add
#   输入:     ['layer3_ff_matmul2.output', 'layer3_ff_bias2']
#   输出:     ['layer3_ff_bias2_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer3_ff_add
#   操作类型: Add
#   输入:     ['layer3_attn_add.output', 'layer3_ff_bias2_add.output']
#   输出:     ['layer3_ff_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_norm1
#   操作类型: LayerNormalization
#   输入:     ['layer3_ff_add.output', 'layer4_ln1_scale', 'layer4_ln1_bias']
#   输出:     ['layer4_norm1.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer4_qkv_matmul
#   操作类型: MatMul
#   输入:     ['layer4_norm1.output', 'layer4_qkv_weight']
#   输出:     ['layer4_qkv_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_qkv_bias_add
#   操作类型: Add
#   输入:     ['layer4_qkv_matmul.output', 'layer4_qkv_bias']
#   输出:     ['layer4_qkv_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_qkv_split
#   操作类型: Split
#   输入:     ['layer4_qkv_bias_add.output']
#   输出:     ['layer4_q.output', 'layer4_k.output', 'layer4_v.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer4_q_reshape
#   操作类型: Reshape
#   输入:     ['layer4_q.output', 'layer4_q_shape']
#   输出:     ['layer4_q_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_q_transpose
#   操作类型: Transpose
#   输入:     ['layer4_q_reshape.output']
#   输出:     ['layer4_q_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer4_k_reshape
#   操作类型: Reshape
#   输入:     ['layer4_k.output', 'layer4_k_shape']
#   输出:     ['layer4_k_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_k_transpose
#   操作类型: Transpose
#   输入:     ['layer4_k_reshape.output']
#   输出:     ['layer4_k_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer4_v_reshape
#   操作类型: Reshape
#   输入:     ['layer4_v.output', 'layer4_v_shape']
#   输出:     ['layer4_v_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_v_transpose
#   操作类型: Transpose
#   输入:     ['layer4_v_reshape.output']
#   输出:     ['layer4_v_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer4_attn_matmul_qk
#   操作类型: MatMul
#   输入:     ['layer4_q_transpose.output', 'layer4_k_transpose.output']
#   输出:     ['layer4_attn_matmul_qk.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_attn_scale
#   操作类型: Mul
#   输入:     ['layer4_attn_matmul_qk.output', 'layer4_scale']
#   输出:     ['layer4_attn_scaled.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_attn_softmax
#   操作类型: Softmax
#   输入:     ['layer4_attn_scaled.output']
#   输出:     ['layer4_attn_softmax.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer4_attn_matmul_v
#   操作类型: MatMul
#   输入:     ['layer4_attn_softmax.output', 'layer4_v_transpose.output']
#   输出:     ['layer4_attn_matmul_v.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_attn_transpose
#   操作类型: Transpose
#   输入:     ['layer4_attn_matmul_v.output']
#   输出:     ['layer4_attn_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer4_attn_reshape
#   操作类型: Reshape
#   输入:     ['layer4_attn_transpose.output', 'layer4_attn_reshape_shape']
#   输出:     ['layer4_attn_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_proj_matmul
#   操作类型: MatMul
#   输入:     ['layer4_attn_reshape.output', 'layer4_proj_weight']
#   输出:     ['layer4_proj_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_proj_bias_add
#   操作类型: Add
#   输入:     ['layer4_proj_matmul.output', 'layer4_proj_bias']
#   输出:     ['layer4_proj_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_attn_add
#   操作类型: Add
#   输入:     ['layer3_ff_add.output', 'layer4_proj_bias_add.output']
#   输出:     ['layer4_attn_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_norm2
#   操作类型: LayerNormalization
#   输入:     ['layer4_attn_add.output', 'layer4_ln2_scale', 'layer4_ln2_bias']
#   输出:     ['layer4_norm2.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer4_ff_matmul1
#   操作类型: MatMul
#   输入:     ['layer4_norm2.output', 'layer4_ff_weight1']
#   输出:     ['layer4_ff_matmul1.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_ff_bias1_add
#   操作类型: Add
#   输入:     ['layer4_ff_matmul1.output', 'layer4_ff_bias1']
#   输出:     ['layer4_ff_bias1_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_ff_relu
#   操作类型: Relu
#   输入:     ['layer4_ff_bias1_add.output']
#   输出:     ['layer4_ff_relu.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_ff_matmul2
#   操作类型: MatMul
#   输入:     ['layer4_ff_relu.output', 'layer4_ff_weight2']
#   输出:     ['layer4_ff_matmul2.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_ff_bias2_add
#   操作类型: Add
#   输入:     ['layer4_ff_matmul2.output', 'layer4_ff_bias2']
#   输出:     ['layer4_ff_bias2_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer4_ff_add
#   操作类型: Add
#   输入:     ['layer4_attn_add.output', 'layer4_ff_bias2_add.output']
#   输出:     ['layer4_ff_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_norm1
#   操作类型: LayerNormalization
#   输入:     ['layer4_ff_add.output', 'layer5_ln1_scale', 'layer5_ln1_bias']
#   输出:     ['layer5_norm1.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer5_qkv_matmul
#   操作类型: MatMul
#   输入:     ['layer5_norm1.output', 'layer5_qkv_weight']
#   输出:     ['layer5_qkv_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_qkv_bias_add
#   操作类型: Add
#   输入:     ['layer5_qkv_matmul.output', 'layer5_qkv_bias']
#   输出:     ['layer5_qkv_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_qkv_split
#   操作类型: Split
#   输入:     ['layer5_qkv_bias_add.output']
#   输出:     ['layer5_q.output', 'layer5_k.output', 'layer5_v.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer5_q_reshape
#   操作类型: Reshape
#   输入:     ['layer5_q.output', 'layer5_q_shape']
#   输出:     ['layer5_q_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_q_transpose
#   操作类型: Transpose
#   输入:     ['layer5_q_reshape.output']
#   输出:     ['layer5_q_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer5_k_reshape
#   操作类型: Reshape
#   输入:     ['layer5_k.output', 'layer5_k_shape']
#   输出:     ['layer5_k_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_k_transpose
#   操作类型: Transpose
#   输入:     ['layer5_k_reshape.output']
#   输出:     ['layer5_k_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer5_v_reshape
#   操作类型: Reshape
#   输入:     ['layer5_v.output', 'layer5_v_shape']
#   输出:     ['layer5_v_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_v_transpose
#   操作类型: Transpose
#   输入:     ['layer5_v_reshape.output']
#   输出:     ['layer5_v_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer5_attn_matmul_qk
#   操作类型: MatMul
#   输入:     ['layer5_q_transpose.output', 'layer5_k_transpose.output']
#   输出:     ['layer5_attn_matmul_qk.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_attn_scale
#   操作类型: Mul
#   输入:     ['layer5_attn_matmul_qk.output', 'layer5_scale']
#   输出:     ['layer5_attn_scaled.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_attn_softmax
#   操作类型: Softmax
#   输入:     ['layer5_attn_scaled.output']
#   输出:     ['layer5_attn_softmax.output']
#   属性:     ['axis: 不支持']
# 节点信息:
#   名称:     layer5_attn_matmul_v
#   操作类型: MatMul
#   输入:     ['layer5_attn_softmax.output', 'layer5_v_transpose.output']
#   输出:     ['layer5_attn_matmul_v.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_attn_transpose
#   操作类型: Transpose
#   输入:     ['layer5_attn_matmul_v.output']
#   输出:     ['layer5_attn_transpose.output']
#   属性:     ['perm: [0, 2, 1, 3]']
# 节点信息:
#   名称:     layer5_attn_reshape
#   操作类型: Reshape
#   输入:     ['layer5_attn_transpose.output', 'layer5_attn_reshape_shape']
#   输出:     ['layer5_attn_reshape.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_proj_matmul
#   操作类型: MatMul
#   输入:     ['layer5_attn_reshape.output', 'layer5_proj_weight']
#   输出:     ['layer5_proj_matmul.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_proj_bias_add
#   操作类型: Add
#   输入:     ['layer5_proj_matmul.output', 'layer5_proj_bias']
#   输出:     ['layer5_proj_bias_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_attn_add
#   操作类型: Add
#   输入:     ['layer4_ff_add.output', 'layer5_proj_bias_add.output']
#   输出:     ['layer5_attn_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_norm2
#   操作类型: LayerNormalization
#   输入:     ['layer5_attn_add.output', 'layer5_ln2_scale', 'layer5_ln2_bias']
#   输出:     ['layer5_norm2.output']
#   属性:     ['epsilon: 9.999999974752427e-07']
# 节点信息:
#   名称:     layer5_ff_matmul1
#   操作类型: MatMul
#   输入:     ['layer5_norm2.output', 'layer5_ff_weight1']
#   输出:     ['layer5_ff_matmul1.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_ff_bias1_add
#   操作类型: Add
#   输入:     ['layer5_ff_matmul1.output', 'layer5_ff_bias1']
#   输出:     ['layer5_ff_bias1_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_ff_relu
#   操作类型: Relu
#   输入:     ['layer5_ff_bias1_add.output']
#   输出:     ['layer5_ff_relu.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_ff_matmul2
#   操作类型: MatMul
#   输入:     ['layer5_ff_relu.output', 'layer5_ff_weight2']
#   输出:     ['layer5_ff_matmul2.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_ff_bias2_add
#   操作类型: Add
#   输入:     ['layer5_ff_matmul2.output', 'layer5_ff_bias2']
#   输出:     ['layer5_ff_bias2_add.output']
#   属性:     []
# 节点信息:
#   名称:     layer5_ff_add
#   操作类型: Add
#   输入:     ['layer5_attn_add.output', 'layer5_ff_bias2_add.output']
#   输出:     ['layer5_ff_add.output']
#   属性:     []
# 节点信息:
#   名称:     ln_final
#   操作类型: LayerNormalization
#   输入:     ['layer5_ff_add.output', 'ln_final_scale', 'ln_final_bias']
#   输出:     ['output0']
#   属性:     ['epsilon: 9.999999974752427e-07']

# ==================================================
# 解析初始化权重信息
# ==================================================
# 初始化权重信息:
#   名称:     layer0_ln1_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer0_ln1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer0_qkv_weight
#   数据类型: FLOAT
#   形状:     [512, 1536]
# 初始化权重信息:
#   名称:     layer0_qkv_bias
#   数据类型: FLOAT
#   形状:     [1536]
# 初始化权重信息:
#   名称:     layer0_proj_weight
#   数据类型: FLOAT
#   形状:     [512, 512]
# 初始化权重信息:
#   名称:     layer0_proj_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer0_q_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer0_k_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer0_v_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer0_scale
#   数据类型: FLOAT
#   形状:     [1]
# 初始化权重信息:
#   名称:     layer0_attn_reshape_shape
#   数据类型: INT64
#   形状:     [3]
# 初始化权重信息:
#   名称:     layer0_ln2_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer0_ln2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer0_ff_weight1
#   数据类型: FLOAT
#   形状:     [2048, 512]
# 初始化权重信息:
#   名称:     layer0_ff_bias1
#   数据类型: FLOAT
#   形状:     [2048]
# 初始化权重信息:
#   名称:     layer0_ff_weight2
#   数据类型: FLOAT
#   形状:     [512, 2048]
# 初始化权重信息:
#   名称:     layer0_ff_bias2
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer1_ln1_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer1_ln1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer1_qkv_weight
#   数据类型: FLOAT
#   形状:     [512, 1536]
# 初始化权重信息:
#   名称:     layer1_qkv_bias
#   数据类型: FLOAT
#   形状:     [1536]
# 初始化权重信息:
#   名称:     layer1_proj_weight
#   数据类型: FLOAT
#   形状:     [512, 512]
# 初始化权重信息:
#   名称:     layer1_proj_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer1_q_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer1_k_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer1_v_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer1_scale
#   数据类型: FLOAT
#   形状:     [1]
# 初始化权重信息:
#   名称:     layer1_attn_reshape_shape
#   数据类型: INT64
#   形状:     [3]
# 初始化权重信息:
#   名称:     layer1_ln2_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer1_ln2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer1_ff_weight1
#   数据类型: FLOAT
#   形状:     [2048, 512]
# 初始化权重信息:
#   名称:     layer1_ff_bias1
#   数据类型: FLOAT
#   形状:     [2048]
# 初始化权重信息:
#   名称:     layer1_ff_weight2
#   数据类型: FLOAT
#   形状:     [512, 2048]
# 初始化权重信息:
#   名称:     layer1_ff_bias2
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer2_ln1_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer2_ln1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer2_qkv_weight
#   数据类型: FLOAT
#   形状:     [512, 1536]
# 初始化权重信息:
#   名称:     layer2_qkv_bias
#   数据类型: FLOAT
#   形状:     [1536]
# 初始化权重信息:
#   名称:     layer2_proj_weight
#   数据类型: FLOAT
#   形状:     [512, 512]
# 初始化权重信息:
#   名称:     layer2_proj_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer2_q_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer2_k_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer2_v_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer2_scale
#   数据类型: FLOAT
#   形状:     [1]
# 初始化权重信息:
#   名称:     layer2_attn_reshape_shape
#   数据类型: INT64
#   形状:     [3]
# 初始化权重信息:
#   名称:     layer2_ln2_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer2_ln2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer2_ff_weight1
#   数据类型: FLOAT
#   形状:     [2048, 512]
# 初始化权重信息:
#   名称:     layer2_ff_bias1
#   数据类型: FLOAT
#   形状:     [2048]
# 初始化权重信息:
#   名称:     layer2_ff_weight2
#   数据类型: FLOAT
#   形状:     [512, 2048]
# 初始化权重信息:
#   名称:     layer2_ff_bias2
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer3_ln1_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer3_ln1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer3_qkv_weight
#   数据类型: FLOAT
#   形状:     [512, 1536]
# 初始化权重信息:
#   名称:     layer3_qkv_bias
#   数据类型: FLOAT
#   形状:     [1536]
# 初始化权重信息:
#   名称:     layer3_proj_weight
#   数据类型: FLOAT
#   形状:     [512, 512]
# 初始化权重信息:
#   名称:     layer3_proj_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer3_q_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer3_k_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer3_v_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer3_scale
#   数据类型: FLOAT
#   形状:     [1]
# 初始化权重信息:
#   名称:     layer3_attn_reshape_shape
#   数据类型: INT64
#   形状:     [3]
# 初始化权重信息:
#   名称:     layer3_ln2_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer3_ln2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer3_ff_weight1
#   数据类型: FLOAT
#   形状:     [2048, 512]
# 初始化权重信息:
#   名称:     layer3_ff_bias1
#   数据类型: FLOAT
#   形状:     [2048]
# 初始化权重信息:
#   名称:     layer3_ff_weight2
#   数据类型: FLOAT
#   形状:     [512, 2048]
# 初始化权重信息:
#   名称:     layer3_ff_bias2
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_ln1_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_ln1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_qkv_weight
#   数据类型: FLOAT
#   形状:     [512, 1536]
# 初始化权重信息:
#   名称:     layer4_qkv_bias
#   数据类型: FLOAT
#   形状:     [1536]
# 初始化权重信息:
#   名称:     layer4_proj_weight
#   数据类型: FLOAT
#   形状:     [512, 512]
# 初始化权重信息:
#   名称:     layer4_proj_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_q_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer4_k_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer4_v_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer4_scale
#   数据类型: FLOAT
#   形状:     [1]
# 初始化权重信息:
#   名称:     layer4_attn_reshape_shape
#   数据类型: INT64
#   形状:     [3]
# 初始化权重信息:
#   名称:     layer4_ln2_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_ln2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer4_ff_weight1
#   数据类型: FLOAT
#   形状:     [2048, 512]
# 初始化权重信息:
#   名称:     layer4_ff_bias1
#   数据类型: FLOAT
#   形状:     [2048]
# 初始化权重信息:
#   名称:     layer4_ff_weight2
#   数据类型: FLOAT
#   形状:     [512, 2048]
# 初始化权重信息:
#   名称:     layer4_ff_bias2
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer5_ln1_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer5_ln1_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer5_qkv_weight
#   数据类型: FLOAT
#   形状:     [512, 1536]
# 初始化权重信息:
#   名称:     layer5_qkv_bias
#   数据类型: FLOAT
#   形状:     [1536]
# 初始化权重信息:
#   名称:     layer5_proj_weight
#   数据类型: FLOAT
#   形状:     [512, 512]
# 初始化权重信息:
#   名称:     layer5_proj_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer5_q_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer5_k_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer5_v_shape
#   数据类型: INT64
#   形状:     [4]
# 初始化权重信息:
#   名称:     layer5_scale
#   数据类型: FLOAT
#   形状:     [1]
# 初始化权重信息:
#   名称:     layer5_attn_reshape_shape
#   数据类型: INT64
#   形状:     [3]
# 初始化权重信息:
#   名称:     layer5_ln2_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer5_ln2_bias
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     layer5_ff_weight1
#   数据类型: FLOAT
#   形状:     [2048, 512]
# 初始化权重信息:
#   名称:     layer5_ff_bias1
#   数据类型: FLOAT
#   形状:     [2048]
# 初始化权重信息:
#   名称:     layer5_ff_weight2
#   数据类型: FLOAT
#   形状:     [512, 2048]
# 初始化权重信息:
#   名称:     layer5_ff_bias2
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     ln_final_scale
#   数据类型: FLOAT
#   形状:     [512]
# 初始化权重信息:
#   名称:     ln_final_bias
#   数据类型: FLOAT
#   形状:     [512]

# 总节点数: 157
# 总初始化器数: 104
