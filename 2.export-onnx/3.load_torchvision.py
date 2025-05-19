import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
import os
import argparse
import torchvision

def get_model(type, dir):
    """
    Select a pre-trained vision model from torchvision and define its ONNX output file path.

    Args:
        type (str): Model type (e.g., 'resnet', 'vgg', 'mobilenet', etc.).
        dir (str): Directory to save the ONNX model.

    Returns:
        tuple: (model, file_path) where model is the PyTorch model and file_path is the ONNX file path.
    """
    if type == "resnet":
        model = torchvision.models.resnet50()
        file = os.path.join(dir, "resnet50.onnx")
    elif type == "vgg":
        model = torchvision.models.vgg11()
        file = os.path.join(dir, "vgg11.onnx")
    elif type == "mobilenet":
        model = torchvision.models.mobilenet_v3_small()
        file = os.path.join(dir, "mobilenet_v3_small.onnx")
    elif type == "efficientnet":
        model = torchvision.models.efficientnet_b0()
        file = os.path.join(dir, "efficientnet_b0.onnx")
    elif type == "efficientnetv2":
        model = torchvision.models.efficientnet_v2_s()
        file = os.path.join(dir, "efficientnet_v2_s.onnx")
    elif type == "regnet":
        model = torchvision.models.regnet_x_1_6gf()
        file = os.path.join(dir, "regnet_x_1_6gf.onnx")
    else:
        raise ValueError(f"Unsupported model type: {type}")
    return model, file

def infer(model, input_tensor):
    """Perform inference using the specified model and input tensor.

    Args:
        model (nn.Module): Pre-trained PyTorch model from torchvision.
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).

    Returns:
        torch.Tensor: Model output tensor.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input_tensor)
    print("Inference output shape:", output.shape)
    return output

def export_norm_onnx(model, file, input_tensor):
    """Export the model to ONNX format, validate, and simplify it.

    Args:
        model (nn.Module): PyTorch model to export.
        file (str): Path to save the ONNX model.
        input_tensor (torch.Tensor): Dummy input tensor for tracing, shape (batch_size, 3, 224, 224).

    Note:
        - do_constant_folding=True may fuse BatchNorm into Conv layers, reducing nodes.
        - ONNX graph may contain Identity nodes due to residual connections (e.g., in ResNet) or
          unfused operations. Use onnx-simplifier to minimize these.
        - Inspect the exported model with Netron to analyze nodes like Identity.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Export model to ONNX format
    torch.onnx.export(
        model=model,              # Model to export
        args=(input_tensor,),     # Dummy input for tracing
        f=file,                   # Output file path
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
    print(f"Finished normal ONNX export to: {file}")

    # Load and validate the ONNX model
    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)  # Check model integrity

    # Simplify the ONNX model using onnx-simplifier
    # This removes redundant nodes (e.g., Identity nodes from residual connections)
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "ONNX simplification failed"
    onnx.save(model_onnx, file)  # Save simplified model
    print(f"Finished ONNX simplification. Model saved to: {file}")

def main(args):
    """Main function to select and export a pre-trained vision model to ONNX.

    Args:
        args: Command-line arguments containing model type and output directory.

    Note:
        - Requires torchvision for pre-trained models. Install via requirements.txt (e.g., torch==2.7.0, torchvision).
        - The exported ONNX graph may include Identity nodes due to residual connections or
          BatchNorm fusion. Use Netron to inspect the graph.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    # Create output directory
    os.makedirs(args.dir, exist_ok=True)  # Ensure directory exists

    # Define input tensor (batch_size=1, channels=3, height=224, width=224)
    input_tensor = torch.rand(1, 3, 224, 224, device='cuda')

    # Get model and output file path
    model, file = get_model(args.type, args.dir)

    # Move model to CUDA
    model.cuda()

    # Perform inference
    infer(model, input_tensor)

    # Export model to ONNX
    export_norm_onnx(model, file, input_tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export pre-trained vision models to ONNX")
    parser.add_argument("-t", "--type", type=str, default="resnet",
                        help="Model type: resnet, vgg, mobilenet, efficientnet, efficientnetv2, regnet")
    parser.add_argument("-d", "--dir", type=str, default="./2.export-onnx/models/",
                        help="Directory to save ONNX models")
    
    opt = parser.parse_args()
    main(opt)
