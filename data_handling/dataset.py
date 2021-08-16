from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split
import numpy as np


class Dataset:
    def __init__(self, data_path: Path):
        """
        Handles all the data
        :param data_path: path to the raw data
        """
        self.data_path = data_path
        self.train = {}
        self.test = {}

    def split(self, test_size: float = 0.3):
        """
        Split the data to test and train
        :param test_size: percent from the whole data. default 30%
        """
        for artist in self.data_path.iterdir():
            artist_paintings = []
            self.train[artist.stem] = []
            self.test[artist.stem] = []
            for painting in artist.iterdir():
                artist_paintings.append(painting)
            train, test = train_test_split(artist_paintings, test_size=test_size)
            self.train[artist.stem].extend(train)
            self.test[artist.stem].extend(test)

    def k_fold(self, k=3, val_size: int = 0.3) -> (Tuple[Path, str], Tuple[Path, str]):
        """
        Generator that creates random folds from the train set
        :param k: number of folds
        :param val_size: size of validation set
        :return: train set and validation set
        """
        for fold in range(k):
            train = []
            val = []
            for artist, paintings in self.train.items():
                v_size = round(len(paintings) * val_size)
                np.random.shuffle(paintings)
                train.extend(paintings[:-v_size])
                val.extend(paintings[-v_size:])

            yield train, val

    def train_one_by_one(self):
        for artist in self.train.keys():
            for image in self.train[artist]:
                yield image, image.parts[-2]

    def test_one_by_one(self):
        for artist in self.test.keys():
            for image in self.test[artist]:
                yield image, image.parts[-2]
