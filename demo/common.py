import argparse

import torch

_AVAILABLE_DEMO_MODELS = (
    'resnet18',
    'resnet32',
    'resnet50',
    'vgg16',
    'vgg19',
    'alexnet',
    'inception_v3',
    # And the efficient_b{i} series
) + tuple(f'efficientnet_b{i}' for i in range(8))


def get_arg_parser() -> argparse.ArgumentParser:
    """
    Retrieve argument parser with required inputs for demo
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=_AVAILABLE_DEMO_MODELS[0], choices=_AVAILABLE_DEMO_MODELS)
    parser.add_argument(
        '--input_shape',
        nargs='+',
        type=int,
        default=[1, 3, 224, 224],
        help='Shape of the input. For vision models, should be of shape ' '(B, C, H, W).',
    )
    return parser


def default_args() -> argparse.Namespace:
    return get_arg_parser().parse_args()


def get_torchvision_model(model_name: str) -> torch.nn.Module:
    if model_name not in _AVAILABLE_DEMO_MODELS:
        raise ValueError('Model is not an available demo models. ' 'The conversion has not been tested')

    # resnet model
    if model_name.startswith('resnet'):
        import torchvision.models.resnet as resnet

        model_fn = getattr(resnet, model_name)
    elif model_name.startswith('vgg'):
        import torchvision.models.vgg as vgg

        model_fn = getattr(vgg, model_name)
    elif model_name.startswith('efficientnet'):
        import torchvision.models.efficientnet as efficientnet

        model_fn = getattr(efficientnet, model_name)
    elif model_name == 'alexnet':
        import torchvision.models.alexnet as alexnet

        model_fn = getattr(alexnet, model_name)
    elif model_name.startswith('inception'):
        from torchvision.models.inception import inception_v3

        model_fn = inception_v3
    return model_fn
