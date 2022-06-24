# pt2keras

A simple PyTorch To Keras Model Converter. 

## Motivation

I needed to export existing models that were developed in PyTorch to edgetpu so that 
we cn utilize the Google TensorBoard and Coral Edge TPUS. Although models already developed 
in TensorFlow were easy to export, I was having difficulty exporting models developed in PyTorch.

If you want to know the root cause, I can add the documentation that I created to the README.md

## Supported Networks

The following networks have been tested and are supported

- EfficientNet
- MobileNetV2
- ResNet

## Installation 

Coming soon ...

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
The example below is how to override the existing `ReLU` operation.

```python
from pt2keras import converter


@converter('Relu', override=True)
def add(onnx_node, input_tensor, *inputs):
    print('overriding ReLU')
    from tensorflow import keras
    return keras.activations.relu(input_tensor)

```

## Updates

The model now supports onnx inputs. 
However, the onnx model must perform operations PyTorch style.
E.g. Model input must be in the form (Batch, Channel, Height, Width).

In a later version, support for other forms will be added ... 

## License

This software is covered by the MIT license.
