import torch
import torch.nn as nn
import torch.onnx
from utils import get_onnx_path

class Model(torch.nn.Module):
    """A simple linear model with custom weights and optional bias."""
    
    def __init__(self, in_features, out_features, weights, bias=False):
        """
        Initialize the linear model with custom weights.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            weights (torch.Tensor): Weight tensor, shape (out_features, in_features).
            bias (bool): Whether to include a bias term. Defaults to False.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)  # Create linear layer
        with torch.no_grad():
            self.linear.weight.copy_(weights)  # Set custom weights

    def forward(self, x):
        """
        Forward pass through the linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (in_features,).

        Returns:
            torch.Tensor: Output tensor of shape (out_features,).
        """
        x = self.linear(x)  # Apply linear transformation
        return x

def infer(weights):
    """Perform inference using the linear model with predefined input and weights."""
    # Define input tensor (1D, 4 features)
    in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    
    # Initialize model with 4 input features, 3 output features
    model = Model(4, 3, weights)
    
    # Run inference and print result
    x = model(in_features)
    print("Result for input [1, 2, 3, 4] is:", x)

def export_onnx(weights):
    """Export the linear model to ONNX format with dynamic batch size."""
    # Define dummy input for ONNX export (batch_size, in_features)
    input = torch.zeros(1, 4)  # Corrected shape to match model input
    
    # Initialize model and set to evaluation mode
    model = Model(4, 3, weights)
    model.eval()  # Set to evaluation mode for stable export

    # Define output path for ONNX model
    output_path = get_onnx_path(__file__, "example_dynamic_shape.onnx")

    # Export model to ONNX format
    torch.onnx.export(
        model=model,              # Model to export
        args=(input,),            # Dummy input for tracing
        f=output_path,            # Output file path
        input_names=["input0"],   # Name for input tensor
        output_names=["output0"], # Name for output tensor
        dynamic_axes={
            'input0': {0: 'batch'},   # Support variable batch size
            'output0': {0: 'batch'}   # Support variable batch size
        },
        opset_version=12,         # ONNX opset version for compatibility
        do_constant_folding=True, # Optimize by folding constants
        verbose=True              # Print export details for debugging
    )
    print(f"Finished ONNX export. Model saved to: {output_path}")

if __name__ == "__main__":
    # Define weight matrix (3 output features x 4 input features)
    weights = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ], dtype=torch.float32)
    
    infer(weights)      # Run inference
    export_onnx(weights) # Export model to ONNX
