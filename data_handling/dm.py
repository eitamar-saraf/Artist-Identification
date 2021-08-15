import shutil
from pathlib import Path

import torch

from data_handling.image_handle import load_image
from model.vgg import VGG


def mkdir_if_not_exists(post_process_path: Path, artist: str):
    if not post_process_path.joinpath(artist).exists():
        post_process_path.joinpath(artist).mkdir(parents=True, exist_ok=True)


def clean_fold(post_process_path: Path):
    shutil.rmtree(post_process_path)


def extract_features_and_save(artist_img: Path, artist: str, device, vgg: VGG, post_process_path:Path):
    painting = load_image(artist_img, device=device)
    painting_features = vgg.get_features(painting)
    mkdir_if_not_exists(post_process_path, artist)
    torch.save(painting_features, post_process_path.joinpath(artist, artist_img.stem))
