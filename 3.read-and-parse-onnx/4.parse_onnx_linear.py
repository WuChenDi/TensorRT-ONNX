import onnx
import os
from onnx import TensorProto
from typing import Dict
from utils import get_onnx_path

def get_tensor_dtype(tensor_type: int) -> str:
    """Map TensorProto data type to a readable string.

    Args:
        tensor_type: Integer representing TensorProto data type (e.g., 1 for FLOAT).

    Returns:
        str: Readable data type (e.g., 'FLOAT', 'INT32').

    Note:
        - Uses dict.get with explicit string conversion to ensure type safety.
        - Unknown types return 'Unknown(tensor_type)' for clarity.
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
    """Load and parse an ONNX model, printing its inputs, outputs, nodes, and initializers."""
    # Define model path
    model_path = get_onnx_path(__file__, "sample-linear.onnx")

    # Load and validate model
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"Failed to load or validate model: {str(e)}")
        return

    # Access graph components
    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node
    inputs = graph.input
    outputs = graph.output

    # Parse inputs
    print("\n" + "="*50)
    print("Parsing Input Information")
    print("="*50)
    for input in inputs:
        input_shape = [d.dim_value if d.dim_value != 0 else None for d in input.type.tensor_type.shape.dim]
        print(f"Input info:\n"
              f"  Name:      {input.name}\n"
              f"  Data Type: {get_tensor_dtype(input.type.tensor_type.elem_type)}\n"
              f"  Shape:     {input_shape}")

    # Parse outputs
    print("\n" + "="*50)
    print("Parsing Output Information")
    print("="*50)
    for output in outputs:
        output_shape = [d.dim_value if d.dim_value != 0 else None for d in output.type.tensor_type.shape.dim]
        print(f"Output info:\n"
              f"  Name:      {output.name}\n"
              f"  Data Type: {get_tensor_dtype(output.type.tensor_type.elem_type)}\n"
              f"  Shape:     {output_shape}")

    # Parse nodes
    print("\n" + "="*50)
    print("Parsing Node Information")
    print("="*50)
    for node in nodes:
        attributes = [f"{attr.name}: {attr.ints or attr.floats or attr.s.decode()}" for attr in node.attribute]
        print(f"Node info:\n"
              f"  Name:      {node.name}\n"
              f"  Op Type:   {node.op_type}\n"
              f"  Inputs:    {node.input}\n"
              f"  Outputs:   {node.output}\n"
              f"  Attributes: {attributes}")

    # Parse initializers
    print("\n" + "="*50)
    print("Parsing Initializer Information")
    print("="*50)
    if not initializers:
        print("No initializers found in the model.")
    for initializer in initializers:
        print(f"Initializer info:\n"
              f"  Name:      {initializer.name}\n"
              f"  Data Type: {get_tensor_dtype(initializer.data_type)}\n"
              f"  Shape:     {list(initializer.dims)}")

if __name__ == "__main__":
    main()

# python3 3.read-and-parse-onnx/4.parse_onnx_linear.py

# ==================================================
# Parsing Input Information
# ==================================================
# Input info:
#   Name:      a
#   Data Type: FLOAT
#   Shape:     [10, 10]
# Input info:
#   Name:      x
#   Data Type: FLOAT
#   Shape:     [10, 10]
# Input info:
#   Name:      b
#   Data Type: FLOAT
#   Shape:     [10, 10]

# ==================================================
# Parsing Output Information
# ==================================================
# Output info:
#   Name:      y
#   Data Type: FLOAT
#   Shape:     [10, 10]

# ==================================================
# Parsing Node Information
# ==================================================
# Node info:
#   Name:      multiply
#   Op Type:   Mul
#   Inputs:    ['a', 'x']
#   Outputs:   ['c']
#   Attributes: []
# Node info:
#   Name:      add
#   Op Type:   Add
#   Inputs:    ['c', 'b']
#   Outputs:   ['y']
#   Attributes: []

# ==================================================
# Parsing Initializer Information
# ==================================================
# No initializers found in the model.
