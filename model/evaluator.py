from pathlib import Path

import torch
import numpy as np

from loss.content_loss import ContentLoss
from loss.style_loss import StyleLoss
from model.vgg import VGG


class Evaluator:

    def __init__(self, device, post_process_path: Path, content_weight: float = 1e-3, style_weight: float = 1e6):
        self.device = device
        self.vgg = VGG(device)
        self.all_features = self._load_all_train(post_process_path)
        self.content_weight = content_weight
        self.style_weight = style_weight

    @staticmethod
    def _load_all_train(post_process_path: Path):
        all_features = {}
        for artist in post_process_path.iterdir():
            all_features[artist.stem] = {}
            for i, painting_features_path in enumerate(artist.iterdir()):
                painting_features = torch.load(painting_features_path)
                all_features[artist.stem][i] = painting_features

        return all_features

    def classify_image(self, painting_img):
        painting_features = self.vgg.get_features(painting_img)
        content_loss = ContentLoss(painting_features['conv5_2'])
        style_loss = StyleLoss(painting_features)

        all_losses = {}
        for artist, paintings_features in self.all_features.items():
            artist_loss = []
            for index, paint_features in paintings_features.items():

                c_loss = content_loss(paint_features['conv5_2'])
                s_loss = 0.0
                for layer in style_loss.style_weights:
                    paint_feature = paint_features[layer]

                    s = style_loss(paint_feature, layer)
                    s_loss += s

                artist_loss.append(self.content_weight * c_loss + self.style_weight * s_loss)
            all_losses[artist] = np.mean(artist_loss)
        return all_losses
