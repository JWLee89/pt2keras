from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_converter_absolute_path() -> str:
    """
    Get the absolute directory path of the converter
    Returns:
        The absolute path of the converter package
    """
    return f'{get_project_root()}/pt2keras/onnx_backend/convert'
