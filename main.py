import argparse

import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Artist identification')
    parser.add_argument('--action', type=str, choices=['train', 'server'], help='Action that you want to preform')

    args = parser.parse_args()

    if args.action == 'train':
        Dataset()
        DataLoader()
        DataDict
        ResNet()
        Adam()
        if classification:
            BCEWithLogits()
        elif representation:
            TripletLoss()

        Solver()
        solver.train()
        solver.eval()
        solver.save_model()

    elif args.action == 'server':

        print(1)

    else:
        print('You somehow bypass the choices constraint'
              'The action you are tying to preform is not exists'
              'The script will exit now')
        exit()
