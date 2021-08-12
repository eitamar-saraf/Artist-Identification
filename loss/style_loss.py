import torch
from torch import nn
import torch.nn.functional as F


class StyleLoss(nn.Module):

    def __init__(self, query_features):
        super(StyleLoss, self).__init__()
        self.query_features = query_features
        self.style_weights = {'conv1_1': 1.5,
                              'conv2_1': 0.80,
                              'conv3_1': 0.25,
                              'conv4_1': 0.25,
                              'conv5_1': 0.25}

    def forward(self, input, layer):
        query_feature = self.gram_matrix(self.query_features[layer])
        refrence_feature = self.gram_matrix(input)
        loss = F.mse_loss(refrence_feature, query_feature)
        return loss * self.style_weights[layer]

    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        gram = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return gram.div(a * b * c * d)
