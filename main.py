import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from model.evaluator import Evaluator
from model.vgg import VGG
from utils.image_handle import load_image

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Artist identification')
    parser.add_argument('--action', type=str, choices=['train', 'eval', 'server'],
                        help='Action that you want to preform')
    parser.add_argument('--train_data_path', type=str, default='raw_data/train/')
    parser.add_argument('--saved_features', type=str, default='post_process_data/')
    parser.add_argument('--predict_data_path', type=str, default='raw_data/test/')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.action == 'train':
        train_data_path = Path(args.train_data_path)
        post_process_path = Path(args.saved_features)
        vgg = VGG(device)

        for artist in train_data_path.iterdir():
            for painting in artist.iterdir():
                painting_img = load_image(painting, device=device)

                painting_features = vgg.get_features(painting_img)
                if not post_process_path.joinpath(artist.stem).exists():
                    post_process_path.joinpath(artist.stem).mkdir(parents=True, exist_ok=True)
                torch.save(painting_features, post_process_path.joinpath(artist.stem, painting.stem))

    if args.action == 'eval':
        post_process_path = Path(args.saved_features)
        predict_data_path = Path(args.predict_data_path)
        evaluator = Evaluator(device, post_process_path)
        counter_true = 0.0
        total_counter = 0.0
        for artist in predict_data_path.iterdir():
            for painting in artist.iterdir():
                total_counter += 1
                painting_img = load_image(painting, device=device)
                scores_with_classes = evaluator.classify_image(painting_img)
                classes = list(scores_with_classes)
                print(f'Real class was: {artist.stem}')
                scores = torch.from_numpy(np.array(list(scores_with_classes.values())))
                prob = F.softmax(-1 * scores, dim=0)
                pred = torch.argmax(prob)
                print(f'Prediction was: {classes[pred]}, confidence was: {prob.max()}')
                if artist.stem == classes[pred]:
                    counter_true += 1
        print(f'Accuracy: {counter_true / total_counter}')

    elif args.action == 'server':

        raise NotImplementedError

    else:
        print('You somehow bypass the choices constraint\n'
              'The action you are tying to preform is not exists\n'
              'The script will exit now')
        exit()
