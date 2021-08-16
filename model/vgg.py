from typing import Dict

import torch
from torchvision import models


class VGG:

    def __init__(self, device: torch.device):
        """
        Holds the model.
        pay attention, the model won't calculate gradients!
        :param device: That we load the data(cpu or gpu)
        """
        self.device = device
        self.model = models.vgg19(pretrained=True).features.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.layers = {'0': 'conv1_1',
                       '5': 'conv2_1',
                       '10': 'conv3_1',
                       '19': 'conv4_1',
                       '30': 'conv5_2',  # content
                       '28': 'conv5_1'}

    def get_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Obtain all the features from self.layers
        :param image: tensor of the image we want to obtain the features
        :return: dict with all the features. each key represent layer
        """

        features = {}
        x = image
        with torch.no_grad():
            for name, layer in self.model._modules.items():
                x = layer(x)
                if name in self.layers:
                    features[self.layers[name]] = x

        return features
