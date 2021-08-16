from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt


def load_image(img_path: Path, device: torch.device, max_size: int = 156) -> torch.Tensor:
    """
    load image from given path.
    This function also resize the image. default 156X156
    also normalize the data like imagenet
    :param img_path: path to the image
    :param device: That we load the data(cpu or gpu)
    :param max_size: of the image
    :return: tensor of the image. shape (1, 3, max_size, max_size)
    """
    image = Image.open(img_path)

    in_transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3, :, :].unsqueeze(0).to(device, torch.float)

    return image


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def image_show(query, style):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(im_convert(query))
    ax1.set_title("Query-Image", fontsize=20)
    ax2.imshow(im_convert(style))
    ax2.set_title("Style-Image", fontsize=20)
    plt.show()
