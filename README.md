# pt2keras

A simple PyTorch To Keras Model Converter. 

## Motivation

I needed to export existing models that were developed in PyTorch to edgetpu so that 
we cn utilize the Google TensorBoard and Coral Edge TPUS. Although models already developed 
in TensorFlow were easy to export, I was having difficulty exporting models developed in PyTorch.

If you want to know the root cause, I can add the documentation that I created to the README.md

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

The steps above are detailed in `demo.py` and the code snippet below

```python

if __name__ == '__main__':
    from copy import deepcopy
    import numpy as np
    import tensorflow as tf
    import torch
    from torchvision.models.efficientnet import efficientnet_b0

    from pt2keras import Pt2Keras

    # Test pt2keras on EfficientNet_b0
    model = efficientnet_b0(pretrained=True).eval()
    height_width = 32

    # Generate dummy inputs
    x_keras = tf.random.normal((1, height_width, height_width, 3))
    # input dimensions for PyTorch are BCHW, whereas TF / Keras default is BHWC
    x_pt = torch.from_numpy(deepcopy(x_keras.numpy())).permute(0, 3, 1, 2)

    print(f'pt shape: {x_pt.shape}, x_keras.shape: {x_keras.shape}')

    # Convert the model
    converter = Pt2Keras(model, x_pt.shape)
    keras_model = converter.convert()

    # Make PT model the same input dimension as Keras
    # If the output is >= 4 dimensional
    pt_output = model(x_pt).cpu().detach().numpy()
    keras_output = keras_model(x_keras).numpy()
    # Mean average diff over all axis

    average_diff = np.mean(np.mean([pt_output, keras_output]))
    print(f'pytorch: {pt_output.shape}')
    print(f'keras: {keras_output.shape}')
    # The differences will be precision errors from multiplication / division
    print(f'Mean average diff: {average_diff}')

    output_is_approximately_equal = np.allclose(pt_output, keras_output, atol=1e-4)
    assert output_is_approximately_equal, f'PyTorch output and Keras output is different. ' \
                                          f'Mean difference: {average_diff}'
```
