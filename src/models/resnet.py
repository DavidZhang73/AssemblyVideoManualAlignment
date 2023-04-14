import torchvision
from torch import nn


def make_resnet50(pretrained=False, freeze=False, feature_dim=1000, keep_last_layer=True):
    model = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT" if pretrained else None)
    if not keep_last_layer:
        model.fc = nn.Identity()
    elif feature_dim != 1000:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=feature_dim, bias=True)
    if freeze:
        for name, param in model.named_parameters():
            if not name.startswith("fc") and not name.startswith("layer4"):
                param.requires_grad = False
    return model
