import torch
import torchvision.transforms as transforms
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = cv2.imread(image_name)
    image = cv2.resize(image, (imsize, imsize))

    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)
