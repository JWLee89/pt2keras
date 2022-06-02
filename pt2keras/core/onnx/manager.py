import logging


class OnnxManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        import onnx

        self.onnx_version = onnx.version.version


if __name__ == '__main__':
    onnx_manager = OnnxManager()
    print(onnx_manager.onnx_version)
