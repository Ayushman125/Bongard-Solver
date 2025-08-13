import torch.nn as nn

class FeatureExtractorDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class AdaptiveClassifierDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, features, params=None):
        if params:
            w = params['fc.weight']; b = params['fc.bias']
            return nn.functional.linear(features, w, b)
        return self.fc(features)
