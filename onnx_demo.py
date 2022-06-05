import torch
import torch.nn as nn
import tensorflow as tf
from torchvision.models.resnet import resnet18

from pt2keras.main import Pt2Keras


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # These can all be found using named_modules() or children()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1), groups=32, dilation=(1, 1),  bias=True),
            nn.SiLU(),
            # nn.ConvTranspose2d(32, 64, (3, 3), (2, 2), padding=(1, 1))
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 16, (1, 1), (2, 2), bias=True),
        #     # nn.BatchNorm2d(16),
        #     # nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
        # )

    def forward(self, X):
        # we can retrieve these operations via .modules(), named_modules(), children(), etc.
        output = self.conv(X)
        return output
        # return output[:, :, 2, 4]


if __name__ == '__main__':
    model = resnet18().eval()
    # model = Model()
    width_height = 32

    x = torch.ones(1, 3, width_height, width_height)

    # Set the model to training mode to get the full computational graph
    # import torch._C as _C
    # TrainingMode = _C._onnx.TrainingMode
    # Convert model
    converter = Pt2Keras(model, x.shape)
    keras_model = converter.convert()
    #
#    pt_output = model(x).permute(0, 2, 3, 1)
    pt_output = model(x)
    if len(pt_output.shape) == 4:
        pt_output = pt_output.permute(0, 2, 3, 1)

    x_tf = tf.ones((1, width_height, width_height, 3))

    print(f'pytorch: {pt_output.shape}, {pt_output}')
    keras_output = keras_model(x_tf)
    print(f'keras: {keras_output.shape}, {keras_output}')

    keras_model.save('model.h5')
