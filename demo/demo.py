"""
@Author Jay Lee

A minimalistic demo on how to use Pt2Keras.
This will convert a target torchvision model into keras.
"""
import tensorflow as tf
from common import default_args, get_torchvision_model

from pt2keras import Pt2Keras

if __name__ == '__main__':
    args = default_args()
    model_name = args.model
    input_shape = args.input_shape

    # Grab mode
    model = get_torchvision_model(model_name).eval()

    # Create pt2keras object
    converter = Pt2Keras(model, input_shape)

    # convert model
    keras_model: tf.keras.Model = converter.convert()

    # Save the model
    keras_model.save('output_model.h5')
