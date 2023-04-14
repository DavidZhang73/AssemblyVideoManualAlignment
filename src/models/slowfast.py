from pytorchvideo.models import hub
from torch import nn


def make_slowfast(pretrained=False, freeze=False, feature_dim=400, keep_last_layer=True):
    model = hub.slowfast_r50(pretrained=pretrained)
    if not keep_last_layer:
        model.blocks[-1].proj = nn.Identity()
    elif feature_dim != 400:
        proj_layer = model.blocks[-1]
        proj_layer.proj = nn.Linear(in_features=proj_layer.proj.in_features, out_features=feature_dim, bias=True)
    if freeze:
        for name, param in model.named_parameters():
            if not name.startswith("blocks.6.proj"):
                param.requires_grad = False
    return model
