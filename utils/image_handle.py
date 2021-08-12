import torch
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt


def load_image(img_path, device, max_size=156):
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
