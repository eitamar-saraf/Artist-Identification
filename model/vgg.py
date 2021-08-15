import torch
from torchvision import models


class VGG:
    def __init__(self, device):
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

    def get_features(self, image):

        features = {}
        x = image
        with torch.no_grad:
            for name, layer in self.model._modules.items():
                x = layer(x)
                if name in self.layers:
                    features[self.layers[name]] = x

        return features
