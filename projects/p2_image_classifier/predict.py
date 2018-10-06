import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

import json

parser = argparse.ArgumentParser(description='Predict What flower the image is')

parser.add_argument('input', action='store',
    type=str,
    help='Image to be predicted on')

parser.add_argument('checkpoint', action='store',
    type = str,
    default='checkpoint.pth',
    help='existing stored checkpoint')

parser.add_argument('--top_k', action='store',
    default=1,
    type=int,
    help='The number of top indice to be predicted')

parser.add_argument('--category_names', action='store',
    default = 'cat_to_name.json',
    type = str,
    help='The class to name mapper json file')

parser.add_argument('--gpu', action='store_true',
    help='Use gpu or not', default=False)


result = parser.parse_args()

input_img = result.input
checkpoint = result.checkpoint
top_k = result.top_k
category_names = result.category_names 
gpu = result.gpu

########################################################################

# CHECK if GPU was passed
if (gpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)


# Checkpoint function
def load_checkpoint(filepath):
#   checkpoint = torch.load(filepath)
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = models.densenet169(pretrained=False)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1664, 800)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(800, 360)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(360, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    # Load model
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['model_class_idx'] 
    
    return model
########################################################################

# load the Model
model = load_checkpoint(checkpoint)

# TODO: Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    pil_image_size = pil_image.size
    aspect_ratio = pil_image_size[0] / pil_image_size[1]
    
    dict_image_size = {value:pos for pos,value in enumerate(pil_image_size)}
    shortest_pos_side = dict_image_size[min(dict_image_size)]
    
    # if width is the shortest (0) then width = 256, height = 256 * aspect ratio,
    # if height is the shortest (1) then width = 256 * 1 / aspect ratio height = 256

    aspect_transforms = [[256, aspect_ratio*256], 
                         [(1/aspect_ratio) * 256, 256]]
        
    pil_image.thumbnail((aspect_transforms[shortest_pos_side][0],
                        aspect_transforms[shortest_pos_side][1]))

    
    left_margin = (pil_image.width-224) / 2
    bottom_margin = (pil_image.height-224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    img = pil_image.crop((left_margin, bottom_margin, right_margin,    
                    top_margin))
    
    img = np.array(img)/255
    means = np.array([0.485, 0.456, 0.406]) 
    stdevs = np.array([0.229, 0.224, 0.225])
    
    np_image = (img - means / stdevs).transpose(2, 0, 1)


    return np_image    


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()

    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    
    image =  torch.from_numpy(img).type(torch.FloatTensor)
    image = image.to(device)
    image = image.unsqueeze_(0)
    output = model.forward(image.float())
    
    # Convert from natural log to probability disbn     
    ps = torch.exp(output)
    
    # Get the top probabilities and their indices   
    prob, indices = ps.topk(topk)
    
    # Get the mapping of the class to index from the model     
    indx_map = model.class_to_idx
    
    # Invert the mapping to be index to class so that we can link 
    # the output indices from forward
    inv_map = {value: key for key, value in indx_map.items()}
    classes = [inv_map[i] for i in indices[0].cpu().numpy()]
    
    return prob[0].cpu().numpy(), classes   

def answer(image_path, model, top):
    probs, classes = predict(image_path, model, topk=top)
    flower_names = [cat_to_name[c] for c in classes]
    return flower_names, probs


predicted_flowers, probs = answer(input_img, model, top_k)
print('My prediction(s): {}'.format(predicted_flowers))
print('My prediction(s): {}'.format(probs))
