import sys
import time
import json
import os
from os import path
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

def process_image(image_path):
    ''' 
        Scales, Crops, Normalizes a PIL image for a PyTorch model,
        returns an Numpy array.
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    image = Image.open(image_path)
    
    resize_size = 256
    final_size = 224
    width, height = image.size
        
    if height > width:
        height = int(height * resize_size / width)
        width = int(resize_size)
    else:
        width = int(width * resize_size / height)
        height = int(resize_size)
        
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    
    # Crop center portion of the image
    x0 = (width - final_size) / 2
    y0 = (height - final_size) / 2
    x1 = x0 + final_size
    y1 = y0 + final_size
    crop_image = resized_image.crop((x0,y0,x1, y1))
    
    # Normalize:
    np_image = np.array(crop_image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    return np_image.transpose(2,0,1)

def load_checkpoint(file_path):
    '''
        Loads a pretrained model from a checkpoint file
    '''
    checkpoint = torch.load(file_path)

    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Architecture \"{}\" not recongized.".format(args.arch))
        parser.print_help()
        sys.exit()

    for x in model.parameters():
        x.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def predict(image_path, model, topk, device):
    ''' 
        Predict the topk classes of an image using a pre-trained model.
    '''
    model.eval()

    np_array = process_image(image_path)
    tensor_in = torch.from_numpy(np_array)

    tensor_in = tensor_in.float() 
    tensor_in = tensor_in.unsqueeze(0)

    model.to(device)
    tensor_in.to(device)

    with torch.no_grad():
      output = model.forward(tensor_in.cuda())  

    output = torch.exp(output)

    topk_probs, topk_indexes = torch.topk(output, topk) 
    topk_probs = topk_probs.tolist()[0]
    topk_indexes = topk_indexes.tolist()[0]
    
    idx_to_cat = {val: key for key, val in model.class_to_idx.items()}
    
    top_cats = [idx_to_cat[index] for index in topk_indexes ]

    return topk_probs, top_cats# , top_labels

if __name__ == '__main__':
    '''
        This script predicts a label based on the model and image given
        as an argument when running the script.
    '''

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path",  
                        help="Image to be classified")
    parser.add_argument("--checkpoint",  
                        help="Checkpoint to load trained model")
    parser.add_argument("--topk",
                        help="NUmber of top-k classes to show",
                        default=5)
    parser.add_argument("--gpu",
                        help="Option for use GPU. CPU used if no specified.",
                        action="store_true")
    parser.add_argument("--category_name",
                        help="File to load with the mapped cat to real names.",
                        default="cat_to_name.json")
    parser.add_argument("-v",
                        "--visual_output",
                        help="Shows the image and a bar chart with the top k classes.",
                        action="store_true")
    args = parser.parse_args()

    if args.image_path == None:
        print("Image path parameter not specified ")
        parser.print_help()
        sys.exit()
    
    if args.checkpoint == None:
        print("Checkpoint file parameter not specified ")
        parser.print_help()
        sys.exit()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on GPU.")
    elif args.gpu and not torch.cuda.is_available():
        print("Error: --gpu was set but cuda device not available.")
        print("Warning: Running on CPU.")
    else:
        device = torch.device("cpu")
        print("Running on CPU.")

    image_path = args.image_path
    check_point_path = args.checkpoint
    top_k = args.topk
    show_image = args.visual_output

    if not path.exists(args.category_name):
        print("Category name file: {} does not exist".format(args.category_name))
        print("Please enter a valid path")
        sys.exit()

    with open(args.category_name, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(check_point_path)

    probs, classes = predict(image_path, model, top_k, device)

    y = [cat_to_name.get(i) for i in classes[::]]
    x = np.array(probs)

    print("Image: {}".format(image_path))
    print("Checkpoint: {}".format(check_point_path))
    print("Probabilies: {}".format(x))
    print("Classes: {}\n".format(y))
    
    if show_image:

        max_index = np.argmax(probs)
        max_probability = probs[max_index]
        label = classes[max_index]
        top_labels = [cat_to_name[cat] for cat in classes ]

        plt.figure(figsize=(6,12))
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)

        image = Image.open(image_path)
        ax1.axis('off')
        ax1.set_title('Flower : ' + cat_to_name[label])
        ax1.imshow(image)

        y_pos = np.arange(5)
        ax2.set_title('Topk=5 Probabilities')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_labels)
        ax2.set_xlabel('Probability')
        ax2.barh(y_pos, probs, align='center')
        ax2.invert_yaxis()

        plt.show()

    print("End of program...")