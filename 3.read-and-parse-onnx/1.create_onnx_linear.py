# Understanding the ONNX model structure
# - ModelProto: Describes the entire model's information
#   - GraphProto: Describes the entire network's information
#     - NodeProto: Describes individual computation nodes (e.g., Conv, Linear)
#     - TensorProto: Describes tensor information, mainly weights
#     - ValueInfoProto: Describes input/output information
#     - AttributeProto: Describes attributes of nodes
# ------------------------------------------------------------------------------

import onnx
from onnx import helper
from onnx import TensorProto
from utils import get_onnx_path

def create_onnx() -> onnx.ModelProto:
    """Create a simple ONNX model for the computation y = a * x + b.

    Returns:
        onnx.ModelProto: The constructed ONNX model.
    """
    # Create input and output tensors (ValueInfoProto)
    # Each tensor is defined with name, type (FLOAT), and shape [10, 10]
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])  # Input tensor a
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])  # Input tensor x
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])  # Input tensor b
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 10])  # Output tensor y

    # Create computation nodes (NodeProto)
    # Mul node: c = a * x
    mul = helper.make_node(
        op_type='Mul',          # Operation type
        inputs=['a', 'x'],      # Input tensors
        outputs=['c'],          # Output tensor
        name='multiply'         # Node name for debugging
    )
    # Add node: y = c + b
    add = helper.make_node(
        op_type='Add',          # Operation type
        inputs=['c', 'b'],      # Input tensors
        outputs=['y'],          # Output tensor
        name='add'              # Node name for debugging
    )

    # Create the computation graph (GraphProto)
    # The graph includes nodes, name, inputs, and outputs
    graph = helper.make_graph(
        nodes=[mul, add],       # List of nodes
        name='sample-linear',   # Graph name
        inputs=[a, x, b],       # Input tensors
        outputs=[y]             # Output tensor
    )

    # Create the model (ModelProto)
    # Specify opset version and IR version for compatibility
    model = helper.make_model(
        graph,
        producer_name='onnx_example',
        opset_imports=[helper.make_operatorsetid('', 15)]  # ONNX opset version 15
    )
    model.ir_version = 8  # Set IR version to 8 for compatibility with most onnxruntime versions

    # Validate the model
    onnx.checker.check_model(model)  # Ensure model is well-formed
    # Print model information
    print("Model is valid. Model information:")
    print(onnx.helper.printable_graph(model.graph))  # Print the graph structure

    # Define output path for ONNX model
    output_path = get_onnx_path(__file__, "sample-linear.onnx")
    onnx.save(model, output_path)  # Save model to file
    print(f"Model saved to: {output_path}")

    return model

if __name__ == "__main__":
    model = create_onnx()  # Create and save the ONNX model
