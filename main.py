import argparse

import torch

from model.vgg import get_vgg, get_features, VGG
from run import get_style_model_and_losses
from utils.image_handle import load_image, image_show

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Artist identification')
    parser.add_argument('--action', type=str, choices=['train', 'server'], help='Action that you want to preform')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.action == 'train':
        style_img = load_image("data/train/claude_monet/Claude_Monet_037.jpg", device=device)
        query_img = load_image(
            "data/train/claude_monet/Claude_Monet_-_The_Magpie_-_Google_Art_Project.jpg", device=device)

        image_show(query_img, style_img)

        vgg = VGG(device)
        style_img_features = vgg.get_features(style_img)
        query_img_features = vgg.get_features(query_img)

        get_style_model_and_losses(cnn=vgg, normalization_mean=cnn_normalization_mean,
                                   normalization_std=cnn_normalization_std, style_img=style_img, content_img=query_img,
                                   content_layers=content_layers_default, style_layers=style_layers_default,
                                   device=device)

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
        print('You somehow bypass the choices constraint\n'
              'The action you are tying to preform is not exists\n'
              'The script will exit now')
        exit()
