import shutil
from pathlib import Path
from typing import Dict

import torch

from data_handling.image_handle import load_image
from model.vgg import VGG


def mkdir_if_not_exists(post_process_path: Path):
    """
    Check if directory id exists.
    If not create it.
    :param post_process_path: The path to the directory
    """
    if not post_process_path.exists():
        post_process_path.mkdir(parents=True, exist_ok=True)


def clean_fold(post_process_path: Path):
    shutil.rmtree(post_process_path)


def save_features(painting_features: Dict[str, torch.Tensor], painting_path: Path, artist: str,
                  post_process_path: Path):
    """
    save the faetures for future use
    :param painting_features: that we want to save
    :param painting_path: the path to the painting that we want to save it features
    :param artist: the artist of the painting
    :param post_process_path:  where you want to save the painting
    """
    mkdir_if_not_exists(post_process_path.joinpath(artist))
    torch.save(painting_features, post_process_path.joinpath(artist, painting_path.stem))


def move_image_to_folder(image: Path, test_data_path: Path):
    mkdir_if_not_exists(test_data_path)
    shutil.copy(str(image.absolute()), test_data_path)
