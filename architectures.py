import torch
import torch.nn as nn


class FinetuneProbeModel(nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module, linear_probe: torch.nn.Module):
        super(FinetuneProbeModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear_probe = linear_probe

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.linear_probe(features)
        return output
    

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, n_input_features, n_classes):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=n_input_features, out_features=n_classes)

    def forward(self, x):

        # single linear feed forward layer
        output = self.linear(x)

        return output

