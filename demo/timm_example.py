"""
@Author Jay Lee
pt2keras also works with models in timm packages.
Although right now, the number of converters are limited, it should be enough
to support quite a lot of computer vision models.

To run this model, first install timm via
pip install timm

python demo.py --model resnet50 input_shape 1 3 224 224
"""
import tensorflow as tf
import torch
from common import default_args
from timm.models.efficientnet import efficientnet_em

try:
    from src.pt2keras import Pt2Keras
except ImportError:
    from pt2keras import Pt2keras

if __name__ == '__main__':
    args = default_args()
    input_shape = args.input_shape

    # Grab mode
    model = efficientnet_em().eval()

    # Create pt2keras object
    converter = Pt2Keras()

    # convert model
    keras_model: tf.keras.Model = converter.convert(model, torch.randn(input_shape))

    # Save the model
    keras_model.save('output_model.h5')
