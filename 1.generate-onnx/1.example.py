import torch
import torch.nn as nn
import torch.onnx
import os

class Model(torch.nn.Module):
    """A simple linear model with custom weights and optional bias."""
    
    def __init__(self, in_features, out_features, weights, bias=False):
        """
        Initialize the linear model with given weights.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            weights (torch.Tensor): Weight tensor of shape (out_features, in_features).
            bias (bool): Whether to include a bias term. Defaults to False.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
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
        x = self.linear(x)
        return x

def infer():
    """Perform inference using the linear model with predefined input and weights."""
    # Define input tensor (1D, 4 features)
    in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    
    # Define weight matrix (3 output features x 4 input features)
    weights = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ], dtype=torch.float32)

    # Initialize model with 4 input features, 3 output features, and custom weights
    model = Model(4, 3, weights)
    
    # Run inference and print result
    x = model(in_features)
    print("result is: ", x)

def export_onnx():
    """Export the linear model to ONNX format."""
    # Create directory for saving the model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Define dummy input for ONNX export (batch_size, in_features)
    input = torch.zeros(1, 4)  # Adjusted to match model's expected input shape
    
    # Define weight matrix (same as in infer)
    weights = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ], dtype=torch.float32)

    # Initialize model
    model = Model(4, 3, weights)
    model.eval()  # Set model to evaluation mode for export

    # Define path for saving the ONNX model
    output_path = os.path.join(model_dir, "example.onnx")

    # Export model to ONNX format
    torch.onnx.export(
        model=model,              # Model to export
        args=(input,),            # Dummy input for tracing
        f=output_path,            # Output file path
        input_names=["input0"],    # Name for input tensor
        output_names=["output0"],  # Name for output tensor
        opset_version=12,         # ONNX opset version (12 for compatibility)
        dynamic_axes={
            'input0': {0: 'batch_size'},   # Allow variable batch size
            'output0': {0: 'batch_size'}   # Allow variable batch size
        },
        do_constant_folding=True,  # Optimize by folding constants
        verbose=True               # Print export details for debugging
    )
    print(f"Finished onnx export. Model saved to: {output_path}")

if __name__ == "__main__":
    infer()       # Run inference
    export_onnx() # Export model to ONNX
