import argparse

import torch
from torchvision import models

from run import get_style_model_and_losses
from utils.image_handle import image_loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Artist identification')
    parser.add_argument('--action', type=str, choices=['train', 'server'], help='Action that you want to preform')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.action == 'train':
        style_img = image_loader("data/claude_monet/Beach_at_Pourville.jpg")
        query_img = image_loader(
            "data/claude_monet/Bridge_Over_a_Pond_of_Water_Lilies,_Claude_Monet_1899.jpg")

        # desired depth layers to compute style/content losses :
        content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        model = models.vgg19(pretrained=True).features.to(device).eval()

        get_style_model_and_losses(cnn=model, normalization_mean=cnn_normalization_mean,
                                   normalization_std=cnn_normalization_std, style_img=style_img, content_img=query_img,
                                   content_layers=content_layers_default, style_layers=style_layers_default, device=device)

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
