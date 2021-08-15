from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split
import numpy as np


class Splitter:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.train = {}
        self.test = {}

    def split(self, ratio: float = 0.3):
        """

        :param ratio:
        :return:
        """
        for artist in self.data_path.iterdir():
            artist_paintings = []
            self.train[artist] = []
            self.test[artist] = []
            for painting in artist.iterdir():
                artist_paintings.append(painting)
            train, test = train_test_split(artist_paintings, test_size=ratio)
            self.train[artist].extend(train)
            self.test[artist].extend(test)

        # TODO save test and train

    def k_fold(self, k=3) -> (Tuple[Path, str], Tuple[Path, str]):
        """

        :param k:
        :return:
        """
        for fold in range(k):
            train = []
            val = []
            for artist, paintings in self.train.items():
                np.random.shuffle(paintings)
                train.extend(paintings[:4])
                val.extend(paintings[4:])

            yield train, val

    def one_by_one(self):
        for artist in self.train.keys():
            for image in self.train[artist]:
                yield image, image.parts[-2]
