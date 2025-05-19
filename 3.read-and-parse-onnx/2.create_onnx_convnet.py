# ------------------------------------------------------------------------------
# Create a basic ConvNet using onnx.helper
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
    """Create an initializer tensor (TensorProto) for weights or biases.

    Args:
        name: Name of the tensor.
        tensor_array: NumPy array containing the tensor data.
        data_type: ONNX data type (default: FLOAT).

    Returns:
        onnx.TensorProto: The initialized tensor.
    """
    initializer = helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist()
    )
    return initializer

def infer_onnx(model: onnx.ModelProto) -> None:
    """Run inference on the ONNX model using ONNX Runtime to verify functionality.

    Args:
        model: The ONNX model to infer.

    Note:
        - Requires onnxruntime and numpy. Install via: pip install onnxruntime numpy
        - Input shape: (1, 3, 64, 64); Output shape: (1, 16, 1, 1).
        - Output is a list of NumPy arrays, with shape attribute returning a tuple (e.g., (1, 16, 1, 1)).
        - session.run may return Sequence[ndarray | SparseTensor | list | dict], but this model outputs ndarray.
        - Ensure onnxruntime supports IR version 8 (requires >= 1.10.0).
    """
    try:
        import onnxruntime as ort

        # Check onnxruntime version
        import pkg_resources
        ort_version = pkg_resources.get_distribution("onnxruntime").version
        # print(f"onnxruntime version: {ort_version}")
        if ort_version < "1.10.0":
            print(f"Warning: onnxruntime {ort_version} may not support IR version {model.ir_version}. Upgrade to >= 1.10.0.")

        # Create random input data
        input_data: ndarray = np.random.randn(1, 3, 64, 64).astype(np.float32)

        # Create inference session
        session = ort.InferenceSession(model.SerializeToString())
        
        # Run inference
        inputs = {"input0": input_data}
        # Expect List[ndarray], validated below
        outputs: List[ndarray] = session.run(None, inputs)  # type: ignore
        
        # Check output and validate type
        if not outputs:
            raise ValueError("Inference produced no outputs.")
        if not isinstance(outputs[0], np.ndarray):
            raise TypeError(f"Expected output type np.ndarray, got {type(outputs[0])}")
        print("Inference output shape:", outputs[0].shape)  # Expected: (1, 16, 1, 1)

    except ImportError:
        print("ONNX Runtime or NumPy not installed. Skipping inference.")
    except Exception as e:
        print(f"Inference failed: {str(e)}. Check onnxruntime version (IR version {model.ir_version}).")

def main() -> None:
    """Create and save a simple ConvNet ONNX model."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define model dimensions
    input_shape = [1, 3, 64, 64]  # [batch, channels, height, width]
    output_shape = [1, 16, 1, 1]  # [batch, channels, height, width]

    # Create input and output tensors (ValueInfoProto)
    input = helper.make_tensor_value_info("input0", TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_shape)

    # Create first Conv node
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

    # Create BatchNorm node
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

    # Create ReLU node
    relu1_node = helper.make_node(
        name="relu1",
        op_type="Relu",
        inputs=["batchNorm1.output"],
        outputs=["relu1.output"],
    )

    # Create GlobalAveragePool node
    global_avg_pool1_node = helper.make_node(
        name="global_avg_pool1",
        op_type="GlobalAveragePool",
        inputs=["relu1.output"],
        outputs=["global_avg_pool1.output"],
    )

    # Create second Conv node
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

    # Create the graph (GraphProto)
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

    # Create the model (ModelProto)
    model = helper.make_model(
        graph,
        producer_name="onnx-sample",
        opset_imports=[helper.make_operatorsetid("", 15)]  # ONNX opset version 15
    )
    model.ir_version = 8  # Set IR version for onnxruntime compatibility

    # Infer shapes and validate model
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    # Save the model
    try:
        # Define output path for ONNX model
        output_path = get_onnx_path(__file__, "sample-convnet.onnx")
        onnx.save(model, output_path)  # Save model to file
        print(f"Congratulations!! Succeed in creating {output_path}")
    except Exception as e:
        print(f"Failed to save model: {str(e)}")

    # Run inference to verify
    infer_onnx(model)

if __name__ == "__main__":
    main()
