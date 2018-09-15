# Yoel Ramos 
# yoelrc88@gmail.com

import sys
import time
import json
import os
import argparse

import torch
from torch import nn
from torch import optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Check packages version
# print('torch version: ', torch.__version__)
# print('torchvision: version', torchvision.__version__)
# print('PIL version: ', Image.PILLOW_VERSION)

def validate(model, device, criterion, dataloader, percent=100):
    model.eval()
    accuracy = 0
    valid_loss = 0
    model.to(device)
    steps = 0
    
    for inputs, labels in iter(dataloader):
          
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output).data 
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
        steps += 1
        status = 100*steps/len(dataloader)
        
        print("", end='\r')
        print("{:3.0f}% done - Loss: {:3.1f} - Accuracy: {:3.0f}%  ".format(
            status, valid_loss/steps, 100*accuracy/steps),end='')
        
        # This allows to do validation in a smaller part of the data
        if steps > (percent / 100.0 * len(dataloader)):
            break

    print("", end='\r')
    return valid_loss/steps, 100*accuracy/steps

def train(model, device, criterion, optimizer, loader_train, loader_valid, epochs, learn_rate):
    
    data_len = len(loader_train)
    validate_every = data_len # see validation 1 time by epoch
    steps = 0
    
    model.train()
    model.to(device)

    print("Starting training:")
    print(" - epochs: {}".format(epochs))
    print(" - learning_rate: {}".format(learn_rate))
    print("Status:")
    
    # Training loop
    for e in range(epochs):
        for inputs, labels in iter(loader_train):
            epoch_status = 100*(steps%data_len)/data_len
            print("", end='\r')
            print("Epoch {:d}/{:d} - {:3.0f}%".format(e+1,epochs,epoch_status),end="")
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)

            model.train()
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            
            if steps % validate_every == 0 or steps==1:
                model.train()
                print()
                valid_loss, accuracy = validate(model, criterion, loader_valid, 10)
                print("Valid Loss: {:.3f} ".format(valid_loss),
                      "Valid Accuracy %: {:.3f}".format(accuracy))

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_dir",  
                        help="Directory of the data", 
                        default=os.getcwd())
    parser.add_argument("-s",
                        "--save_dir",  
                        help="Directory to save the checkpoint", 
                        default="./")
    parser.add_argument("-a",
                        "--arch",
                        help="Type of architecture. Accepted \"vgg19\" or \"alexnet\".",
                        default="vgg19")
    parser.add_argument("-lr",
                        "--learning_rate",
                        help="Learning rate used for training.",
                        default=0.003,
                        type=float)
    parser.add_argument("-m",
                        "--momentum",
                        help="Momentum used for training.",
                        default=0.9,
                        type=float)
    parser.add_argument("-hu",
                        "--hidden_units",
                        help="Number of units in hidden layers.",
                        default=12595,
                        type=int)
    parser.add_argument("-hl",
                        "--hidden_layers_num",
                        help="Number of hidden layers.",
                        default=1,
                        type=int)
    parser.add_argument("-dp",
                        "--drop_out",
                        help="Drop out.",
                        default=0.6,
                        type=float)
    parser.add_argument("-e",
                        "--epochs",
                        help="Number of epochs used for training.",
                        default=3,
                        type=int)
    parser.add_argument("--gpu",
                        help="Option for use GPU. CPU used if no specified.",
                        action="store_true" )
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir

    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    # Defining Transforms for input data
    data_transforms_train = transforms.Compose(
        [transforms.RandomRotation(30, False, True),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transforms_valid = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transforms_test = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    # Load the datasets with ImageFolder
    image_dataset_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    image_dataset_valid = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
    image_dataset_test  = datasets.ImageFolder(test_dir, transform=data_transforms_test)

    # Using the image datasets and the transforms, define the dataloaders
    dataloader_train  = torch.utils.data.DataLoader(image_dataset_train, batch_size=64, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(image_dataset_valid, batch_size=32)
    dataloader_test  = torch.utils.data.DataLoader(image_dataset_test, batch_size=32)

    class_to_idx = image_dataset_train.class_to_idx

    # Number of images in the loaders
    train_num = len(image_dataset_train.imgs)
    print("Train dataset count:     ", train_num)
    valid_num = len(image_dataset_valid.imgs)
    print("Validation dataset count:", valid_num)
    test_num = len(image_dataset_test.imgs)
    print("Test dataset count:      ",test_num)

    # Loading mapping from category label to category name.
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # From args
    learning_rate = args.learning_rate
    momentum = args.momentum
    hidden_units = args.hidden_units
    hidden_layers_num = args.hidden_layers_num
    drop_out = args.drop_out
    epochs = args.epochs
    use_gpu = args.gpu
    output_size = 102 # Number of flower classes

    # Checking for GPU device usage or CPU
    # device = "cpu" #TODO: remove if no needed
    if use_gpu and torch.cuda.is_available():
        device = torch.device("gpu")
        print("Running on GPU.")
    elif use_gpu and not torch.cuda.is_available():
        print("Error: --gpu was set but cuda device not available.")
        print("Warning: Running on CPU.")
    else:
        device = torch.device("cpu")
        print("Running on CPU.")

    # Creating model with specific architecture
    if args.arch == "vgg19":
        model = models.vgg19(pretrained=True)
        input_size = 25088
    elif args.arch == "alexnet":
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif args.arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_size = 9216
    else:
        print("Model \"{}\" not recongized.".format(args.arch))
        args.print_help()
        sys.exit()

    # freezing parameters of features
    for param in model.parameters():
        param.requires_grad = False

    net_layers = nn.ModuleList([nn.Linear(input_size, hidden_units)])
    net_layers.extend([nn.Linear(hidden_units, hidden_units) for i in range(1, hidden_layers_num-1)])
    net_layers.append(nn.Linear(hidden_units, output_size))

    # Classifier params
    params = OrderedDict()
    for i in range(len(net_layers)):
        if i == 0:
            # update input layer
            params.update({'fc{}'.format(i + 1): net_layers[i]})
        else:
            # update hidden layers
            params.update({'relu{}'.format(i): nn.ReLU()})
            params.update({'drop{}'.format(i): nn.Dropout(p=drop_out)})
            params.update({'fc{}'.format(i + 1): net_layers[i]})

    # update output layer
    params.update({'output': nn.LogSoftmax(dim=1)})
    print("Classifier structure:")
    print(params)

    model.classifier = nn.Sequential(params)
    model.to(device)
    model.train()

    # Loss function
    criterion = nn.NLLLoss()
    # Optimizer
    optimizer = optim.SGD(
        model.classifier.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    # Start training
    # train(model, device, criterion, optimizer, dataloader_train,
    #       dataloader_valid, epochs, learning_rate)

    # Save checkpoint
    checkpoint_filename = "checkpoint-" + time.strftime("%Y%m%d-%H%M") + ".pth"

    if save_dir:
        save_dir_file = save_dir + checkpoint_filename
    else:
        save_dir_file = checkpoint_filename

    # Select model checkpoint parameters to include in file
    checkpoint = {  'input_size': input_size,
                    'output_size': output_size,
                    'epochs': epochs,
                    'arch': args.arch,
                    'hidden_units': hidden_units,
                    'hidden_layers_num': hidden_layers_num,
                    'learning_rate': learning_rate,
                    'class_to_idx': class_to_idx,
                    'optimizer_dict': optimizer.state_dict(),
                    'classifier': model.classifier,
                    'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir_file)