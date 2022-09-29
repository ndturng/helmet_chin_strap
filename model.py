import torch.nn as nn
import torchvision.models as models

from repvgg import create_RepVGG_A0
from utils import load_checkpoint

MODEL_PATH = '.../RepVGG-A0-train.pth'

def build_model_repvgg(pretrained=False, deploy=False, fine_tune=False, num_classes=3):
    model = create_RepVGG_A0(deploy)
    
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        # Load pretrained weight
        load_checkpoint(model, MODEL_PATH)
    elif not pretrained:
        fine_tune=True
        print('[INFO]: Not loading pre-trained weights')
    
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad=True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad=False

    # Change the final layer
    num_ftrs = model.linear.in_features
    model.linear = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    return model

def build_model_effi(pretrained=True, fine_tune=False, num_classes=3):
    
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights')
        model = models.efficientnet_b0(weights=None)
        fine_tune=True
        
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad=True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad=False

    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model
