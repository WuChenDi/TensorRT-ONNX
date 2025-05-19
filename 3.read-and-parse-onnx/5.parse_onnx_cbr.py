import torch
import torch.nn as nn
import torch.onnx
import onnx
import numpy as np
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

def parse_onnx(model: onnx.ModelProto) -> None:
    """Parse and print ONNX model structure.

    Args:
        model: The ONNX model to parse.

    Note:
        - Parses inputs, outputs, and nodes from model.graph.
        - Prints name, data type, shape, and node attributes.
    """
    graph = model.graph
    inputs = graph.input
    outputs = graph.output
    nodes = graph.node

    print("\n" + "="*50)
    print("Parsing Input Information")
    print("="*50)
    for input in inputs:
        input_shape = [d.dim_value if d.dim_value != 0 else None for d in input.type.tensor_type.shape.dim]
        print(f"Input info:\n"
              f"  Name:      {input.name}\n"
              f"  Data Type: {get_tensor_dtype(input.type.tensor_type.elem_type)}\n"
              f"  Shape:     {input_shape}")

    print("\n" + "="*50)
    print("Parsing Output Information")
    print("="*50)
    for output in outputs:
        output_shape = [d.dim_value if d.dim_value != 0 else None for d in output.type.tensor_type.shape.dim]
        print(f"Output info:\n"
              f"  Name:      {output.name}\n"
              f"  Data Type: {get_tensor_dtype(output.type.tensor_type.elem_type)}\n"
              f"  Shape:     {output_shape}")

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

def read_weight(item: onnx.TensorProto) -> None:
    """Read and print ONNX initializer weight information.

    Args:
        item: The TensorProto object containing weight data.

    Note:
        - Prints name, data type, and shape of the initializer.
    """
    shape = item.dims
    data  = np.frombuffer(item.raw_data, dtype=np.float32).reshape(shape)
    print("\n" + "="*50)
    print("parse weight data")
    print("="*50)
    print(f"Initializer info:\n"
          f"  Name:      {item.name}\n"
          f"  Data Type: {get_tensor_dtype(item.data_type)}\n"
          f"  Shape:     {list(item.dims)}")

class Model(torch.nn.Module):
    """A simple CNN model with Conv2d, BatchNorm2d, and LeakyReLU.

    Architecture:
        input (1, 3, 5, 5)
          |
        Conv2d (in=3, out=16, kernel=3)
          |
        BatchNorm2d (num_features=16)
          |
        LeakyReLU
          |
        output (1, 16, 3, 3)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(num_features=16)
        self.act1  = nn.LeakyReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

def export_norm_onnx() -> None:
    """Export the PyTorch model to ONNX format.

    Note:
        - Exports a CBR model with input shape (1, 3, 5, 5).
        - Saves to './models/sample-cbr.onnx' with opset_version 15.
        - Input name: 'input0', output name: 'output0'.
    """
    try:
        """Export the convolutional model to ONNX format."""
        # Create input and model
        input = torch.rand(1, 3, 5, 5)
        model = Model()
        model.eval()

        # Define output path for ONNX model
        output_path = get_onnx_path(__file__, "sample-cbr.onnx")

        # Export to ONNX
        torch.onnx.export(
            model         = model,
            args          = (input,),
            f             = output_path,
            input_names   = ["input0"],
            output_names  = ["output0"],
            opset_version = 15
        )
        print(f"Finished exporting ONNX model to {output_path}")
    except Exception as e:
        print(f"Failed to export ONNX model: {str(e)}")

def main() -> None:
    """Main function to export and parse an ONNX model."""
    # Export the model to ONNX
    export_norm_onnx()

    # Load and parse the ONNX model
    try:
        model_path = get_onnx_path(__file__, "sample-cbr.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = onnx.load_model(model_path)
        onnx.checker.check_model(model)
        
        # Parse model structure
        parse_onnx(model)

        # Parse initializers
        print("\n" + "="*50)
        print("Parsing Initializer Information")
        print("="*50)
        initializers = model.graph.initializer
        if not initializers:
            print("No initializers found in the model.")
        for item in initializers:
            read_weight(item)
            
    except Exception as e:
        print(f"Failed to load or parse model: {str(e)}")

if __name__ == "__main__":
    main()

# python3 3.read-and-parse-onnx/5.parse_onnx_cbr.py 
# Finished exporting ONNX model to /home/wudi/work/github/WuChenDi/TensorRT-ONNX/3.read-and-parse-onnx/models/sample-cbr.onnx

# ==================================================
# Parsing Input Information
# ==================================================
# Input info:
#   Name:      input0
#   Data Type: FLOAT
#   Shape:     [1, 3, 5, 5]

# ==================================================
# Parsing Output Information
# ==================================================
# Output info:
#   Name:      output0
#   Data Type: FLOAT
#   Shape:     [1, 16, 3, 3]

# ==================================================
# Parsing Node Information
# ==================================================
# Node info:
#   Name:      /conv1/Conv
#   Op Type:   Conv
#   Inputs:    ['input0', 'onnx::Conv_12', 'onnx::Conv_13']
#   Outputs:   ['/conv1/Conv_output_0']
#   Attributes: ['dilations: [1, 1]', 'group: ', 'kernel_shape: [3, 3]', 'pads: [0, 0, 0, 0]', 'strides: [1, 1]']
# Node info:
#   Name:      /act1/LeakyRelu
#   Op Type:   LeakyRelu
#   Inputs:    ['/conv1/Conv_output_0']
#   Outputs:   ['output0']
#   Attributes: ['alpha: ']

# ==================================================
# Parsing Initializer Information
# ==================================================

# ==================================================
# parse weight data
# ==================================================
# Initializer info:
#   Name:      onnx::Conv_12
#   Data Type: FLOAT
#   Shape:     [16, 3, 3, 3]

# ==================================================
# parse weight data
# ==================================================
# Initializer info:
#   Name:      onnx::Conv_13
#   Data Type: FLOAT
#   Shape:     [16]
