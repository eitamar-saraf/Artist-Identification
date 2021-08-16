import argparse

import torch

from run import k_fold, train

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Artist identification')
    parser.add_argument('--action', type=str, choices=['kfold', 'train', 'server'],
                        help='Action that you want to preform')
    parser.add_argument('--train_data_path', type=str, default='data/raw_data/')
    parser.add_argument('--saved_features', type=str, default='data/post_process_data/')
    parser.add_argument('--test_data_path', type=str, default='data/test_data/')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.action == 'kfold':
        k_fold(args, device)

    elif args.action == 'train':
        train(args, device)

    elif args.action == 'server':

        raise NotImplementedError

    else:
        print('You somehow bypass the choices constraint\n'
              'The action you are tying to preform is not exists\n'
              'The script will exit now')
        exit()
