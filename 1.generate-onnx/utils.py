import os
from typing import Optional

def ensure_extension(model_name: str, extension: str = ".onnx") -> str:
    """
    Ensure the file name has the specified extension.

    Args:
        model_name (str): The file name
        extension (str): The desired extension, defaults to '.onnx'

    Returns:
        str: File name with the specified extension

    Raises:
        ValueError: If model_name is empty
    """
    if not model_name:
        raise ValueError("Model name cannot be empty")
    if not model_name.endswith(extension):
        return f"{model_name}{extension}"
    return model_name

def get_onnx_path(file_path: str, model_name: str, base_dir: Optional[str] = None) -> str:
    """
    Get the save path for an ONNX model.

    Args:
        file_path (str): Path of the current file, typically __file__
        model_name (str): Model file name, e.g., 'model.onnx'
        base_dir (Optional[str]): Custom base directory, defaults to None (uses current file's directory)

    Returns:
        str: Full path for the ONNX model

    Raises:
        ValueError: If file_path or model_name is invalid
        OSError: If directory creation fails

    Examples:
        >>> path = get_onnx_path(__file__, "model.onnx")
        >>> path = get_onnx_path(__file__, "model.onnx", "custom_models")
    """
    if not file_path or not os.path.exists(os.path.dirname(os.path.abspath(file_path))):
        raise ValueError("Invalid or non-existent file path")
    
    model_name = ensure_extension(model_name)
    
    base_path = os.path.dirname(os.path.abspath(file_path))
    if base_dir:
        model_dir = base_dir if os.path.isabs(base_dir) else os.path.join(base_path, base_dir)
    else:
        model_dir = os.path.join(base_path, "models")
    
    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {model_dir}: {e}")
    
    return os.path.normpath(os.path.join(model_dir, model_name))
