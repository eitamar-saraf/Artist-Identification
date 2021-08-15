import argparse
from pathlib import Path

import torch

from data_handling.dm import clean_fold, extract_features_and_save
from data_handling.splitter import Splitter
from model.evaluator import Evaluator
from model.vgg import VGG

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Artist identification')
    parser.add_argument('--action', type=str, choices=['train', 'eval', 'server'],
                        help='Action that you want to preform')
    parser.add_argument('--train_data_path', type=str, default='raw_data/')
    parser.add_argument('--saved_features', type=str, default='post_process_data/')
    parser.add_argument('--predict_data_path', type=str, default='raw_data/test/')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.action == 'train':
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

        for image, artist in splitter.one_by_one():
            extract_features_and_save(image, artist, device, vgg, post_process_path)

    elif args.action == 'server':

        raise NotImplementedError

    else:
        print('You somehow bypass the choices constraint\n'
              'The action you are tying to preform is not exists\n'
              'The script will exit now')
        exit()
