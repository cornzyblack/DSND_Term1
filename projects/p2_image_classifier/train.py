import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot
import seaborn as sb

import json
import argparse

parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('data_dir', action='store',
	help='Directory of Images for training')

parser.add_argument('--save_dir', action='store',
	default='./',
	type=str,
	help='Directory to save checkpoint')

parser.add_argument('--arch', action='store',
	default = 'densenet169',
	type = str,
	help='Model architecture to use for transfer learning')

parser.add_argument('--learning_rate', action='store',
	type=float,
	default=0.0005,
	help='The learning rate for the model')

parser.add_argument('--hidden_units', action='append',
	dest='hidden_units',
	type=int,
	default=[800, 360],
	help='The number of hidden units')

parser.add_argument('--epochs', action='store',
	default=5,
	type=int,
	help='the number of epochs to train for')

parser.add_argument('--gpu', action='store_true',
	help='Use gpu or not', default=True)


########################################################################

# Read in all Arguments passed
result = parser.parse_args()

data_dir = result.data_dir
save_dir = result.save_dir
arch = result.arch
lr = result.learning_rate
hidden_units = result.hidden_units
epochs = result.epochs
save_dir = result.save_dir
gpu = result.gpu
########################################################################


# CHECK if GPU was passed
if (gpu):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device("cpu")


########################################################################
# Set the Data directory for train, valid and test
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

########################################################################
# TRANSFORMATIONS ON DATA
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(p=0.65),
                                      transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406),
                                                           (0.229, 0.224, 0.225))])

                                       
data_transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                      (0.229, 0.224, 0.225))])

########################################################################

# LOAD DATASETS
train_datasets = datasets.ImageFolder(train_dir, transform=train_transform)
valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms)


trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
########################################################################

# GET MODELS
# model_list = {'densenet121': models.densenet121(pretrained=True),
#               'densenet161': models.densenet161(pretrained=True),
#               'densenet169': models.densenet169(pretrained=True),
#               'densenet201': models.densenet201(pretrained=True),
#               'vgg11': models.vgg11(pretrained=True),
#               'vgg13': models.vgg13(pretrained=True),
#               'vgg16': models.vgg16(pretrained=True),
#               'vgg19': models.vgg19(pretrained=True)
#               }
model_list = {'densenet169': models.densenet169(pretrained=True),
              'vgg19': models.vgg19(pretrained=True)}

# check if the architecture is available in the list
# If available get it from the model_list and initialize 
# it in model
if (arch in model_list):
	model = model_list[arch]
# architecture is not in model_list then use the default densenet169
else:
	model = model_list['densenet169']
########################################################################

# CLASSIFIER
layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
modules = [nn.Linear(h1, h2) for h1, h2 in layer_sizes]
relu_layers = [nn.ReLU() for i in np.arange(len(modules))]
module_names = ['fc' + str(i+2) for i in np.arange(len(modules))]
relu_names = ['relu' for i in np.arange(len(relu_layers))]
complete_module = []
complete_relu = []
for i in zip(module_names, modules):
    complete_module.append(i)

for i in zip(relu_names, relu_layers):
    complete_relu.append(i)
final_module = []

input_layer = [('fc1', nn.Linear(1664, hidden_units[0])), ('relu', nn.ReLU())]
for i,x in zip(complete_module, complete_relu):
    final_module.append(i)
    final_module.append(x)

output_layer = [('fc'+ str(len(hidden_units)+1), nn.Linear(hidden_units[-1], 102)), ('output', nn.LogSoftmax(dim=1))] 
tot_modules = input_layer + final_module + output_layer

for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict(tot_modules))

model.classifier = classifier
# ##################################################################

def validation(testloader, model, criterion):    
    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device) 
        outputs = model(images)
        ps = torch.exp(outputs)
        
        test_loss = criterion(outputs, labels).item()
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
#     print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))
    return test_loss, accuracy 
########################################################################

# TRAINING

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr =lr)
steps = 0
running_loss = 0
print_every = 40


model.to(device)
for e in range(epochs):
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:            
            
            model.eval()
            with torch.no_grad():
                test_loss, accuracy = validation(validloader, model, criterion)
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.4f}".format(running_loss/print_every),
                  "Test Loss: {:.4f}".format(test_loss/len(testloader)),
                   "Test Accuracy: {:.4f}".format(accuracy/len(testloader)))
            running_loss = 0

            model.train()
########################################################################

# VALIDATION ON TEST SET
with torch.no_grad():
    _, test_accuracy = check_accuracy_on_test(testloader, model, criterion)
    print('{:.4f}%'.format(test_accuracy))

########################################################################
# CHECKPOINT SAVE

# Get the class to Index for the model
model.class_to_idx = train_datasets.class_to_idx

checkpoint = {'model_class_idx': model.class_to_idx,
              'state_dict': model.state_dict(), 
              'optimizer': optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
########################################################################