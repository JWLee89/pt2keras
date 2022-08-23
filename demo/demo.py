"""
@Author Jay Lee

A minimalistic demo on how to use Pt2Keras.
This will convert a target torchvision model into keras.
To run the script, type in the following:

python demo.py --model resnet50 input_shape 1 3 224 224 --debug
"""
import logging

import tensorflow as tf
import torch
from common import default_args, get_torchvision_model

try:
    from src.pt2keras import Pt2Keras
except ImportError:
    from pt2keras import Pt2keras

if __name__ == '__main__':
    args = default_args()
    model_name = args.model
    input_shape = args.input_shape

    # Grab model
    model = get_torchvision_model(model_name)(pretrained=False).eval()

    # Create pt2keras object
    converter = Pt2Keras()
    if args.debug:
        converter.set_logging_level(logging.DEBUG)

    # convert model
    keras_model: tf.keras.Model = converter.convert(model, torch.randn(input_shape), strict=True)

    # Save the model
    keras_model.save('output_model.h5')
