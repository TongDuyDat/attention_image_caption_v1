from torchvision import models
from torch import nn
import torch
import numpy as np
from torchsummary import summary

class VGG16(nn.Module):
    def __init__(self) -> None:
        super(VGG16, self).__init__()
        self.model = self.load_vgg()
        
    def load_vgg(self):
        vgg = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        # Discard fully connected layer, we are only interested in the features
        layers = list(vgg.children())[:-1]
        return nn.Sequential(*layers[:-1])
    def forward(self, x):
        x = self.model(x)
        return x

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer = 1, batch_first = True, bidirectional = False) -> None:
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layer, batch_first = batch_first, bidirectional = bidirectional)
        for name, param in self.gru.named_parameters():
            if 'bias_ih' in name:
                torch.nn.init.ones_(param)
            elif 'bias_hh' in name:
                torch.nn.init.zeros_(param)
            elif 'weight_ih' in name:
                torch.nn.init.ones_(param)
            elif 'weight_hh' in name:
                torch.nn.init.zeros_(param)
    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        return out, hidden