from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from sklearn.metrics import classification_report
from scipy.special import softmax
from data_handling.image_handle import load_image
from loss.content_loss import ContentLoss
from loss.style_loss import StyleLoss

from model.vgg import VGG


class Evaluator:

    def __init__(self, vgg: VGG, device: torch.device, post_process_path: Path, content_weight: float = 1e-2,
                 style_weight: float = 1e5):
        """
        This class handles all the classification
        :param vgg: the model that extract features
        :param device: That we load the data(cpu or gpu)
        :param post_process_path: path to features directory
        :param content_weight: how much weight the content should receive
        :param style_weight: how much weight the style should receive
        """
        self.device = device
        self.vgg = vgg
        self.all_features = self._load_all_train(post_process_path)
        self.classes = self.get_classes(post_process_path)
        self.content_weight = content_weight
        self.style_weight = style_weight

    @staticmethod
    def _load_all_train(post_process_path: Path) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        """
        loads all features ahead of time.
        to save I/O in infer
        :param post_process_path: path to features directory
        :return: a dict that contains all artists.
        each artist contains all his painting
        each painting contain features from specified layers
        """
        all_features = {}
        for artist in post_process_path.iterdir():
            all_features[artist.stem] = {}
            for i, painting_features_path in enumerate(artist.iterdir()):
                painting_features = torch.load(painting_features_path)
                all_features[artist.stem][i] = painting_features

        return all_features

    def classify_image(self, painting: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Calculate the score of the painting for each class.
        By using style and content loss
        :param painting: tensor of the painting
        :return: score of each class
        """
        painting_features = self.vgg.get_features(painting)
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

                artist_loss.append((self.content_weight * c_loss + self.style_weight * s_loss).detach().cpu().numpy())
            all_losses[artist] = np.mean(artist_loss)
        return all_losses

    def classify_images(self, paintings: List[Path]):
        """
        classify a list of images.
        also prints report on the predictions of that list
        :param paintings: that we want to classify
        """
        y_true = []
        y_pred = []

        for painting in paintings:
            artist = painting.parts[-2]
            painting = load_image(painting, device=self.device)
            scores_with_classes = self.classify_image(painting)
            prob = self.score_to_prob(scores_with_classes)
            pred = np.argmax(prob)
            print(f'scores: {scores_with_classes}')
            print(f'True class: {artist}')
            print(f'prediction {self.classes[pred]}')
            print(f'confidence: {prob[pred]}')
            y_true.append(artist)
            y_pred.append(self.classes[pred])
        print(classification_report(y_true, y_pred, zero_division=1))

    @staticmethod
    def score_to_prob(score: Dict[str, np.ndarray]) -> np.ndarray:
        scores = np.array(list(score.values()))
        prob = softmax(-1 * scores, axis=0)
        return prob

    @staticmethod
    def get_classes(post_process_path: Path) -> List[str]:
        """
        :param post_process_path: path to features directory
        :return: list of classes
        """
        classes = []
        for artist in post_process_path.iterdir():
            classes.append(artist.stem)
        return classes
