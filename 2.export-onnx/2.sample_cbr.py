import torch
import torch.nn as nn
import torch.onnx
import os

class Model(nn.Module):
    """A simple convolutional block with Conv2d, BatchNorm2d, and ReLU activation."""
    
    def __init__(self):
        """
        Initialize the convolutional block.
        
        The block consists of:
        - Conv2d: 3 input channels, 16 output channels, 3x3 kernel
        - BatchNorm2d: Normalizes the 16 output channels
        - ReLU: Applies non-linear activation
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)  # 3x3 convolution
        self.bn1   = nn.BatchNorm2d(num_features=16)  # Batch normalization for 16 channels
        self.act1  = nn.ReLU()  # ReLU activation

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 16, height-2, width-2).
        """
        x = self.conv1(x)  # Apply convolution
        x = self.bn1(x)    # Apply batch normalization
        x = self.act1(x)   # Apply ReLU activation
        return x

def infer():
    """Perform inference using the convolutional model with a random input."""
    # Define input tensor (batch_size=1, channels=3, height=5, width=5)
    input_tensor = torch.rand(1, 3, 5, 5)
    
    # Initialize model
    model = Model()
    
    # Run inference and print output shape
    output = model(input_tensor)
    print("Inference output shape:", output.shape)

def export_norm_onnx():
    """Export the convolutional model to ONNX format."""
    # Create directory for saving the model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Define dummy input for ONNX export (batch_size, channels, height, width)
    input_tensor = torch.rand(1, 3, 5, 5)
    
    # Initialize model and set to evaluation mode
    model = Model()
    model.eval()  # Set to evaluation mode for stable export

    # Define output path for ONNX model
    output_path = os.path.join(model_dir, "sample-cbr.onnx")

    # Export model to ONNX format
    # Note: During export with do_constant_folding=True, BatchNorm2d may be fused with Conv2d
    # into a single Conv operator for optimization, which is why BatchNorm may not appear
    # in the ONNX graph.
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
    print(f"Finished ONNX export. Model saved to: {output_path}")

if __name__ == "__main__":
    infer()            # Run inference
    export_norm_onnx() # Export model to ONNX
