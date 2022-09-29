import torch
import cv2
import numpy as np
import glob as glob
import os
import torch.nn as nn
import torchvision.models as models

from utils import load_checkpoint
from repvgg import create_RepVGG_A0
from torchvision import transforms

# Constants.
DATA_PATH = '.../inputs/data/test/undetermined'
# MODEL_PATH = '.../repvgg_pt50_tuning_steplr.pth'
MODEL_PATH = '.../efficientnet_pt50_tuning_steplr.pth'
IMAGE_SIZE = 128
device = ('cuda' if torch.cuda.is_available() else 'cpu')


# Class names.
class_names = ['no_chin_strap', 'undetermined', 'with_chin_strap']

# Load the trained model.

##RepVGG
# model = create_RepVGG_A0(deploy=False)
# # Change the final layer
# num_ftrs = model.linear.in_features
# model.linear = nn.Linear(num_ftrs, 3)
# checkpoint = torch.load(MODEL_PATH, map_location=device)
# print('Loading trained model weights...')
# model.load_state_dict(checkpoint['model_state_dict'])

#Efficientnet
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(in_features=1280, out_features=3)
checkpoint = torch.load(MODEL_PATH, map_location=device)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Get all the test image paths.
all_image_paths = glob.glob(f'{DATA_PATH}/*')
# Iterate over all the images and do forward pass.
# len(all_image_paths)
count=0
for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    gt_class_name = image_path.split(os.path.sep)[-2]  #.split('.')[0]
    image_name = image_path.split(os.path.sep)[-1]
    # Read the image and create a copy.
    image = cv2.imread(image_path)
    orig_image = image.copy()
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    
    # Forward pass throught the image.
    outputs = model(image)
    outputs = outputs.detach().numpy()
    pred_class_name = class_names[np.argmax(outputs[0])]
    # print(f'GT: {gt_class_name} - image name: {image_name} - Pred: {pred_class_name.lower()}')
    
    if pred_class_name.lower() != gt_class_name:
        count+=1
        print(f'GT: {gt_class_name} - image name: {image_name} - Pred: {pred_class_name.lower()}')
        
acc = count/len(all_image_paths)  

model_name = MODEL_PATH.split(os.path.sep)[-1]
folder_name = DATA_PATH.split(os.path.sep)[-1]

print('-'*50)
print('Model name:', model_name)
print('Folder test:', folder_name)
print('Total images predict:',len(all_image_paths))
print('Images predict wrong:',count)
print('Acc:', 1-acc)
