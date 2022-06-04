import torch

import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.flatten = nn.Flatten(1)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)

        x = self.model.avgpool(x)
        x = self.flatten(x)

        x = self.model.classifier(x)

        return x

    def forward(self, X):
        return self._forward_impl(X)
