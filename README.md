# pt2keras

A simple PyTorch To Keras Model Converter. 

## Motivation

I needed to export existing models that were developed in PyTorch to edgetpu so that 
we cn utilize the Google TensorBoard and Coral Edge TPUS. Although models already developed 
in TensorFlow were easy to export, I was having difficulty exporting models developed in PyTorch.

This project was designed to export PyTorch models to TensorFlow while maintaining the ability to 
upload the model to the EdgeTPU without running into errors.

## Supported Networks

The following networks have been tested and are supported

- EfficientNet
- MobileNetV2
- ResNet
- AlexNet
- Inception_v3 (warning: converted model shows relatively larger distance (Network output value does not fall within atol=1e-4))
- Vgg
- GoogleNet

## Installation 

The package can be installed via the following command. 

```shell

# -U for upgrading existing packages

pip install -U pt2keras

```

Afterwards, try importing the library using the following command:

```shell
from pt2keras import Pt2Keras
```

If it works without any errors, then the package has been successfully `installed`. 
Afterwards, check out the `demo/demo.py` source code for examples on how to use pt2keras.

## How to use

First, import the module

```python
from pt2keras import Pt2Keras
```

Afterwards, we proceed with the following steps: 

1. Define model to convert in PyTorch
2. Convert the model into Keras
3. Perform inference
4. Have a coffee and compare raw outputs. Yee!

For more information, check out the examples inside `demo`. To run the demo, type in the following: 

1. Resnet18 demo

```shell
cd demo 
python demo.py
```

2. Custom PyTorch model demo

```shell
cd demo 
python custom_pytorch_demo.py
# For available arguments, type in the following
# python custom_pytorch_demo.py -h
```

## FAQ

Question: What should I do if I get the following error?

```shell

Traceback (most recent call last):
  File "---", line 90, in <module>
    keras_model = converter.convert()
  File "---", line 78, in convert
    return self.graph._convert()
  File "---", line 233, in _convert
    raise ValueError('Failed to convert model. The following operations are currently unsupported: '
ValueError: Failed to convert model. The following operations are currently unsupported: AveragePool
```

Answer: This means that the `AveragePool` operator is currently not supported.
The framework can be extended without modifying the source code by adding the converter using the following decorator.

```python
from pt2keras import converter

# Update the Relu onnx operator converter
@converter('Relu', override=True)
def add(onnx_node, input_tensor, *inputs):
    print('overriding ReLU')
    from tensorflow import keras
    return keras.activations.relu(input_tensor)
```

The example below is how to override the existing `ReLU` operation.
If the override flag is not provided and the `operator` is already implemented, we will get the following error: 

```shell
Traceback (most recent call last):
  File "---", line 50, in <module>
    @converter('Relu')
  File "----", line 270, in converter
    raise DuplicateOperatorConverterError(f'Converter for "{onnx_op}" already exists ...')
pt2keras.core.onnx.convert.common.DuplicateOperatorError: Converter for "Relu" already exists ...
```

## Updates

`pt2keras` now supports onnx inputs. 
However, the onnx model must perform operations PyTorch style.
E.g. Model input must be in the form (Batch, Channel, Height, Width).

Dynamic batch support was recently added and can be viewed under `demo/custom_pytorch_demo.py`

## License

This software is covered by the MIT license.
