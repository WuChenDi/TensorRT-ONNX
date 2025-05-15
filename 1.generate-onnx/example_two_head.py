import torch
import torch.nn as nn
import torch.onnx
import os

class Model(torch.nn.Module):
    """A linear model with two parallel linear layers, each with custom weights."""
    
    def __init__(self, in_features, out_features, weights1, weights2, bias=False):
        """
        Initialize the model with two linear layers and custom weights.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features per linear layer.
            weights1 (torch.Tensor): Weights for the first linear layer, shape (out_features, in_features).
            weights2 (torch.Tensor): Weights for the second linear layer, shape (out_features, in_features).
            bias (bool): Whether to include bias terms. Defaults to False.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features, bias)  # First linear layer
        self.linear2 = nn.Linear(in_features, out_features, bias)  # Second linear layer
        with torch.no_grad():
            self.linear1.weight.copy_(weights1)  # Set weights for first layer
            self.linear2.weight.copy_(weights2)  # Set weights for second layer

    def forward(self, x):
        """
        Forward pass through both linear layers.

        Args:
            x (torch.Tensor): Input tensor of shape (in_features,).

        Returns:
            tuple: Outputs from both linear layers, each of shape (out_features,).
        """
        x1 = self.linear1(x)  # Output from first linear layer
        x2 = self.linear2(x)  # Output from second linear layer
        return x1, x2

def infer(weights1, weights2):
    """Perform inference using the two-head linear model."""
    # Define input tensor (1D, 4 features)
    in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    
    # Initialize model with 4 input features, 3 output features per layer
    model = Model(4, 3, weights1, weights2)
    
    # Run inference and print results
    x1, x2 = model(in_features)
    print("Inference results:")
    print("Output 1:", x1)
    print("Output 2:", x2)

def export_onnx(weights1, weights2):
    """Export the two-head linear model to ONNX format."""
    # Create directory for saving the model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Define dummy input for ONNX export (batch_size, in_features)
    input = torch.zeros(1, 4)  # Corrected shape to match model input
    
    # Initialize model and set to evaluation mode
    model = Model(4, 3, weights1, weights2)
    model.eval()  # Set to evaluation mode for stable export

    # Define output path for ONNX model
    output_path = os.path.join(model_dir, "example_two_head.onnx")

    # Export model to ONNX format
    torch.onnx.export(
        model=model,              # Model to export
        args=(input,),            # Dummy input for tracing
        f=output_path,            # Output file path
        input_names=["input0"],   # Name for input tensor
        output_names=["output0", "output1"],  # Names for output tensors
        opset_version=12,         # ONNX opset version for compatibility
        dynamic_axes={
            'input0': {0: 'batch_size'},   # Support variable batch size
            'output0': {0: 'batch_size'},  # Support variable batch size
            'output1': {0: 'batch_size'}   # Support variable batch size
        },
        do_constant_folding=True,  # Optimize by folding constants
        verbose=True               # Print export details for debugging
    )
    print(f"Finished ONNX export. Model saved to: {output_path}")

if __name__ == "__main__":
    # Define weight matrices (3 output features x 4 input features)
    weights1 = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ], dtype=torch.float32)
    weights2 = torch.tensor([
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ], dtype=torch.float32)
    
    infer(weights1, weights2)  # Run inference
    export_onnx(weights1, weights2)  # Export model to ONNX
