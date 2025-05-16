import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
import os

class Model(nn.Module):
    """A convolutional neural network with two conv blocks, adaptive pooling, and a linear head."""
    
    def __init__(self):
        """
        Initialize the model with two conv blocks, adaptive pooling, and a linear layer.

        The model consists of:
        - Conv2d (3->16, 3x3) + BatchNorm2d + ReLU
        - Conv2d (16->64, 5x5) + BatchNorm2d + ReLU
        - AdaptiveAvgPool1d to reduce spatial dimensions
        - Linear layer (64->10) for classification
        """
        super().__init__()
        self.conv1   = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # 3x3 conv
        self.bn1     = nn.BatchNorm2d(num_features=16)  # Batch norm for 16 channels
        self.act1    = nn.ReLU()  # ReLU activation
        self.conv2   = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, padding=2)  # 5x5 conv
        self.bn2     = nn.BatchNorm2d(num_features=64)  # Batch norm for 64 channels
        self.act2    = nn.ReLU()  # ReLU activation
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Adaptive pooling to 1D
        self.head    = nn.Linear(in_features=64, out_features=10)  # Linear layer for output

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10).
        """
        x = self.conv1(x)  # Apply first convolution
        x = self.bn1(x)    # Apply batch normalization
        x = self.act1(x)   # Apply ReLU activation
        x = self.conv2(x)  # Apply second convolution
        x = self.bn2(x)    # Apply batch normalization
        x = self.act2(x)   # Apply ReLU activation
        # Flatten spatial dimensions (H, W) to a single dimension
        # Note: torch.flatten(x, 2, 3) generates shape->slice->concat->reshape nodes in ONNX
        x = torch.flatten(x, 2, 3)  # (B, C, H, W) -> (B, C, H*W)
        x = self.avgpool(x)         # (B, C, H*W) -> (B, C, 1)
        x = torch.flatten(x, 1)     # (B, C, 1) -> (B, C)
        x = self.head(x)            # (B, C) -> (B, 10)
        return x

def infer():
    """Perform inference using the model with a random input."""
    # Define input tensor (batch_size=1, channels=3, height=64, width=64)
    input_tensor = torch.rand(1, 3, 64, 64)
    
    # Initialize model
    model = Model()
    
    # Run inference and print output shape
    output = model(input_tensor)
    print("Inference output shape:", output.shape)

def export_norm_onnx():
    """Export the model to ONNX format and validate it."""
    # Create directory for saving the model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Define dummy input for ONNX export (batch_size, channels, height, width)
    input_tensor = torch.rand(1, 3, 64, 64)
    
    # Initialize model and set to evaluation mode
    model = Model()
    model.eval()  # Set to evaluation mode for stable export

    # Define output path for ONNX model
    output_path = os.path.join(model_dir, "sample-reshape.onnx")

    # Export model to ONNX format
    # Note: torch.flatten generates shape->slice->concat->reshape nodes in ONNX.
    # Using onnx-simplifier can merge these into a single Flatten or Reshape node.
    # BatchNorm layers may also be folded into Conv layers with do_constant_folding=True.
    torch.onnx.export(
        model=model,              # Model to export
        args=(input_tensor,),     # Dummy input for tracing
        f=output_path,            # Output file path
        input_names=["input0"],   # Name for input tensor
        output_names=["output0"], # Name for output tensor
        opset_version=15,         # ONNX opset version for compatibility
        dynamic_axes={
            'input0': {0: 'batch'},   # Support dynamic batch size
            'output0': {0: 'batch'}   # Support dynamic batch size
        },
        do_constant_folding=True, # Optimize by folding constants (e.g., BatchNorm into Conv)
        verbose=True              # Print export details for debugging
    )
    
    # Load and validate the ONNX model
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)  # Check model integrity
    
    # Simplify the ONNX model using onnx-simplifier
    # This can reduce complex nodes (e.g., shape->slice->concat->reshape) into simpler ones
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "ONNX simplification failed"
    onnx.save(model_onnx, output_path)  # Save simplified model
    print(f"Finished ONNX export and simplification. Model saved to: {output_path}")

if __name__ == "__main__":
    infer()            # Run inference
    export_norm_onnx() # Export model to ONNX
