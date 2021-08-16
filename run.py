import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from data_handling.dm import extract_features_and_save, clean_fold, mkdir_if_not_exists
from data_handling.image_handle import load_image
from data_handling.splitter import Splitter
from model.evaluator import Evaluator
from model.vgg import VGG


def k_fold(args, device):
    train_data_path = Path(args.train_data_path)
    post_process_path = Path(args.saved_features)

    vgg = VGG(device)

    splitter = Splitter(train_data_path)
    splitter.split()

    for fold, (train, val) in enumerate(splitter.k_fold()):
        print(f'---------------Fold Number {fold + 1}---------------')
        for image in train:
            artist = image.parts[-2]
            extract_features_and_save(image, artist, device, vgg, post_process_path)

        evaluator = Evaluator(vgg, device, post_process_path)
        evaluator.classify_images(val)
        clean_fold(post_process_path)


def move_image_to_folder(image: Path, test_data_path: Path):
    mkdir_if_not_exists(test_data_path)
    shutil.copy(str(image.absolute()), test_data_path)


def train(args, device):
    train_data_path = Path(args.train_data_path)
    post_process_path = Path(args.saved_features)
    test_data_path = Path(args.test_data_path)

    vgg = VGG(device)

    splitter = Splitter(train_data_path)
    splitter.split()

    for image, artist in splitter.train_one_by_one():
        extract_features_and_save(image, artist, device, vgg, post_process_path)

    evaluator = Evaluator(vgg, device, post_process_path)
    y_true = []
    y_pred = []
    for image, artist in splitter.test_one_by_one():
        painting = load_image(image, device)
        score = evaluator.classify_image(painting)
        prob = evaluator.score_to_prob(score)
        pred = np.argmax(prob)
        print(f'Real Class: {artist}')
        print(f'Predicted Class: {evaluator.classes[pred]}')
        print(f'Confidence of Prediction: {prob[pred]}')
        print(f'Confidence of All Classes: {prob}')
        move_image_to_folder(image, test_data_path)
        y_true.append(artist)
        y_pred.append(evaluator.classes[pred])
    print(classification_report(y_true, y_pred, zero_division=1))
