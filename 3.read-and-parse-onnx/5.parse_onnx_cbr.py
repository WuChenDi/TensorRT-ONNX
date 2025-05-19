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

def read_weight(initializer: onnx.TensorProto) -> None:
    """Read and print ONNX initializer weight information.

    Args:
        initializer: The TensorProto object containing weight data.

    Note:
        - Prints name, data type, and shape of the initializer.
    """
    shape = initializer.dims
    data  = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(shape)
    print("\n" + "="*50)
    print("parse weight data")
    print("="*50)
    print(f"Initializer info:\n"
          f"  Name:      {initializer.name}\n"
          f"  Data Type: {get_tensor_dtype(initializer.data_type)}\n"
          f"  Shape:     {list(shape)}\n"
          f"  Data:      \n{data}")

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
#   Data:      
# [[[[-3.06196301e-03  6.21563308e-02 -1.91435754e-01]
#    [-1.84016109e-01  1.86813265e-01  1.07991070e-01]
#    [-1.65803179e-01 -3.74087058e-02 -1.29399106e-01]]

#   [[ 6.20945320e-02  8.47334191e-02  1.02695234e-01]
#    [-1.10583469e-01  2.22384669e-02 -1.35495141e-01]
#    [ 1.52631268e-01  1.69662282e-01  1.25505820e-01]]

#   [[ 1.12754628e-01 -1.88507549e-02  7.46666640e-02]
#    [ 9.17801559e-02  4.70173210e-02 -9.64725167e-02]
#    [-4.59092781e-02  6.75247684e-02 -8.18658918e-02]]]


#  [[[-1.48675397e-01 -1.49901214e-04 -7.83883780e-02]
#    [-1.55972734e-01 -6.94293827e-02 -1.69046983e-01]
#    [ 1.08049415e-01 -1.17127135e-01  5.16503118e-02]]

#   [[-9.05297399e-02  1.39457092e-01 -9.01867375e-02]
#    [ 1.92421615e-01 -1.39869265e-02 -1.40341550e-01]
#    [-2.35402975e-02 -1.67762954e-02  1.73232317e-01]]

#   [[-1.83034569e-01 -9.10378993e-02 -8.05706233e-02]
#    [ 1.60462141e-01 -1.41580418e-01 -9.26839933e-02]
#    [-1.44994602e-01 -4.82615419e-02 -4.59374040e-02]]]


#  [[[-1.36101812e-01 -1.62852138e-01  8.31933413e-03]
#    [ 9.56139714e-02  1.48784027e-01  1.31567210e-01]
#    [ 7.99648017e-02 -7.77960708e-03 -5.28526381e-02]]

#   [[ 1.60463597e-03  1.10024720e-01  1.61302209e-01]
#    [-1.63187787e-01 -4.93264981e-02 -5.46747856e-02]
#    [-1.39150292e-01 -1.11680999e-02 -1.79715827e-02]]

#   [[ 2.95453388e-02  7.21238032e-02 -9.47735459e-02]
#    [ 1.42004639e-01  4.44904268e-02  2.50895750e-02]
#    [-9.88349915e-02  1.81370527e-01 -8.31424817e-02]]]


#  [[[-1.23419138e-02  2.29078606e-02  1.63726687e-01]
#    [ 1.55978814e-01  1.79665893e-01  5.26439808e-02]
#    [ 3.64800245e-02 -5.89739718e-02 -4.20446284e-02]]

#   [[ 1.56685319e-02  4.56444658e-02 -7.24631101e-02]
#    [ 4.68848981e-02 -7.81717673e-02  1.16900243e-02]
#    [ 1.75265089e-01  2.15300731e-02  8.27690586e-02]]

#   [[ 6.21208437e-02 -1.14361122e-01  2.91689802e-02]
#    [-1.52927592e-01  2.75539272e-02 -8.48385841e-02]
#    [ 1.60369650e-01  7.62740970e-02 -1.35777920e-01]]]


#  [[[-5.92304133e-02 -1.30737543e-01  5.80418482e-02]
#    [-1.07463375e-01  1.03624985e-01  8.35429728e-02]
#    [ 1.79466829e-01  1.64044961e-01  1.43069848e-01]]

#   [[ 1.52812630e-01 -1.57802239e-01  1.28128886e-01]
#    [-1.18804723e-01  1.19576260e-01 -9.78157520e-02]
#    [-1.41451553e-01  1.05353601e-01 -1.42847911e-01]]

#   [[-1.11954309e-01  8.56509954e-02  1.43224731e-01]
#    [-1.37108117e-01 -1.17208876e-01  1.43571094e-01]
#    [ 1.55263424e-01 -1.31631285e-01 -1.68402433e-01]]]


#  [[[-8.47239196e-02  1.66126668e-01  1.46574885e-01]
#    [ 1.84703365e-01 -6.17717616e-02  1.21680930e-01]
#    [ 8.83700550e-02 -5.93826547e-02  1.42976478e-01]]

#   [[-2.91705634e-02 -8.20182711e-02 -6.54427111e-02]
#    [-1.22452736e-01 -4.93850000e-02  9.29059759e-02]
#    [ 1.58964582e-02  5.62821738e-02 -1.58519059e-01]]

#   [[-5.33194542e-02  5.53683825e-02 -8.68204981e-02]
#    [ 4.31518219e-02 -6.41670823e-02  8.36080611e-02]
#    [-1.75140575e-01 -5.84986880e-02 -8.03108737e-02]]]


#  [[[ 9.23693702e-02 -1.29851177e-01  1.75668150e-01]
#    [-1.84269413e-01  1.81010798e-01 -9.07387584e-02]
#    [ 1.16807632e-01 -9.10986736e-02 -1.47859812e-01]]

#   [[-7.64034688e-02  1.65964678e-01 -1.22771434e-01]
#    [ 1.21716969e-01 -1.80098370e-01  1.70184761e-01]
#    [-1.57885134e-01  1.25141367e-02  7.46025145e-02]]

#   [[-1.79025888e-01 -1.07715018e-01 -3.58554833e-02]
#    [ 7.68217817e-02  1.14045806e-01  1.72402903e-01]
#    [-1.39911979e-01  1.09141469e-01 -1.61081508e-01]]]


#  [[[-7.98461884e-02  4.23581079e-02  7.20399097e-02]
#    [ 1.73063666e-01  1.32240251e-01 -1.25131786e-01]
#    [-1.04436167e-01 -4.06170115e-02  1.20130524e-01]]

#   [[ 2.17380635e-02 -1.91864416e-01  1.25209615e-01]
#    [ 8.95695388e-02  1.73436217e-02  1.91222623e-01]
#    [-1.06257647e-01 -1.39395356e-01  3.52723747e-02]]

#   [[-5.83365820e-02  2.28331182e-02 -2.51789540e-02]
#    [-8.83329138e-02 -1.02397241e-01 -1.01355940e-01]
#    [-9.24363136e-02  1.90901294e-01 -5.02866954e-02]]]


#  [[[-2.65880357e-02  1.21183284e-01  1.52092159e-01]
#    [-4.16069031e-02  1.51780337e-01 -7.79238418e-02]
#    [ 6.27122596e-02 -1.77300498e-01  5.86141534e-02]]

#   [[-1.01052538e-01  1.68520957e-01 -1.05200671e-01]
#    [-9.74047109e-02 -1.79569460e-02  1.78390145e-01]
#    [-8.33945442e-03 -9.86929834e-02  1.63937494e-01]]

#   [[ 8.00273847e-03 -1.48628354e-01 -5.80476299e-02]
#    [-1.01960383e-01  9.61161852e-02  2.92902496e-02]
#    [-1.20158404e-01 -5.45397289e-02  1.78115085e-01]]]


#  [[[ 1.84862599e-01 -2.66911350e-02  2.55024787e-02]
#    [ 9.39355344e-02 -1.17303558e-01 -1.00315675e-01]
#    [-5.07255271e-03  3.44132297e-02 -7.91806057e-02]]

#   [[ 6.70313835e-02 -6.84216917e-02 -1.90894157e-01]
#    [-3.08636390e-02 -1.54394791e-01 -6.02883240e-03]
#    [-1.34356376e-02 -1.71791524e-01  1.05865728e-02]]

#   [[ 1.16140090e-01  1.08564071e-01  6.56476095e-02]
#    [-2.86818352e-02 -1.05009554e-02  2.80318968e-02]
#    [-1.25546813e-01 -2.03984734e-02  6.25342056e-02]]]


#  [[[ 9.86190587e-02  5.59292622e-02 -1.42287478e-01]
#    [ 1.29136786e-01 -1.04650922e-01 -3.48071824e-03]
#    [ 1.84914947e-01  8.52498189e-02 -3.62036228e-02]]

#   [[ 1.84925690e-01 -1.82066053e-01 -5.44984788e-02]
#    [-7.45510384e-02  5.65823428e-02 -1.26897961e-01]
#    [ 1.13266893e-01 -1.49244145e-01  1.41723499e-01]]

#   [[-5.36771864e-02  3.01680453e-02 -1.72904775e-01]
#    [ 1.90988407e-01 -1.40234038e-01  1.58779085e-01]
#    [ 7.92085528e-02 -1.75979018e-01 -1.38219714e-01]]]


#  [[[-1.55176014e-01 -1.53926350e-02 -6.45097867e-02]
#    [ 1.14896744e-01 -8.77518654e-02  1.56321570e-01]
#    [-1.50032714e-01 -1.45091623e-01 -3.67631502e-02]]

#   [[-1.36413788e-02 -1.86511055e-01 -1.85760975e-01]
#    [ 4.15669829e-02 -1.32488102e-01  6.13099933e-02]
#    [-7.38375708e-02 -1.39847353e-01  5.58314379e-03]]

#   [[-1.14293531e-01  4.96500218e-03 -5.65900281e-02]
#    [ 2.13200189e-02 -1.79802001e-01  1.40942559e-01]
#    [-1.43048391e-01 -4.33108099e-02  1.71397477e-01]]]


#  [[[-7.38865742e-03  4.35208641e-02  7.27052316e-02]
#    [ 5.42559847e-02  1.08522333e-01 -1.87084690e-01]
#    [-9.02487040e-02  1.77085713e-01  4.27709445e-02]]

#   [[-1.30073473e-01 -5.67559451e-02  4.75582853e-02]
#    [-6.42352225e-03  1.08703762e-01  1.31634936e-01]
#    [ 2.11089086e-02 -1.73180059e-01 -9.21376571e-02]]

#   [[ 1.55372351e-01 -5.03808968e-02  9.09883231e-02]
#    [ 8.90039057e-02 -1.20588809e-01 -9.24986899e-02]
#    [-4.49099168e-02  1.73994198e-01  1.00709327e-01]]]


#  [[[ 7.67685547e-02  8.51858314e-03  1.52586028e-01]
#    [ 7.67854899e-02 -6.98818192e-02  6.53519332e-02]
#    [ 1.41204506e-01  1.27303094e-01 -9.68374982e-02]]

#   [[-1.87298894e-01 -4.00362425e-02  2.68152729e-02]
#    [-8.05661976e-02 -2.50158161e-02 -1.69073623e-02]
#    [ 1.49486214e-02  1.28146350e-01 -1.53760523e-01]]

#   [[ 7.16845393e-02  5.89116402e-02 -1.01339974e-01]
#    [-3.86922956e-02  1.18388310e-01 -9.73389819e-02]
#    [-1.93051845e-02  6.95695579e-02  1.86958820e-01]]]


#  [[[-1.12972390e-02  8.19739476e-02 -1.81629106e-01]
#    [ 1.03577770e-01  1.24884233e-01  1.20866068e-01]
#    [ 1.46105453e-01 -1.72950387e-01  1.70390069e-01]]

#   [[ 1.67816192e-01 -1.11334257e-01  1.79966226e-01]
#    [-7.83785135e-02  1.82909921e-01 -2.56210882e-02]
#    [-1.11707799e-01 -4.76784781e-02 -3.12337577e-02]]

#   [[ 1.81516111e-01 -9.40925479e-02  4.78736907e-02]
#    [ 1.11996636e-01 -6.88272640e-02  1.09160282e-01]
#    [ 6.75535351e-02  1.22847766e-01  7.82523863e-03]]]


#  [[[ 5.08868089e-03  1.23754079e-02  1.23571903e-01]
#    [-1.22546710e-01  6.67971233e-03  8.84671658e-02]
#    [ 1.03731647e-01  2.26819310e-02 -5.76533563e-02]]

#   [[-1.57020725e-02  2.69478071e-03  1.43665418e-01]
#    [ 1.85538173e-01  7.72490501e-02  1.77944496e-01]
#    [-1.85273528e-01 -9.84296296e-03 -4.83621396e-02]]

#   [[-3.33876163e-02  9.33746248e-02  1.77166954e-01]
#    [-6.24352805e-02  6.03407957e-02 -1.28927335e-01]
#    [-1.48248941e-01  1.75742969e-01 -1.16887623e-02]]]]

# ==================================================
# parse weight data
# ==================================================
# Initializer info:
#   Name:      onnx::Conv_13
#   Data Type: FLOAT
#   Shape:     [16]
#   Data:      
# [-0.01775997 -0.02957525 -0.17650232  0.01794155  0.04129646 -0.11261131
#   0.02004712 -0.07418653 -0.16775425 -0.15231086  0.00492288  0.03283929
#  -0.0515746   0.1435806  -0.09321461 -0.13102406]
