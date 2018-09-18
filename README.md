# AI Programming with Python at Udacity 
Files from taking the course [**AI Programming with Python**](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) at **Udacity** 

# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='https://github.com/udacity/aipnd-project/blob/master/assets/Flowers.png?raw=true' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

### Imports


```python
import time
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

print('torch version: ', torch.__version__)
print('torchvision: version', torchvision.__version__)
print('PIL version: ', Image.PILLOW_VERSION)
```

    torch version:  0.4.0
    torchvision: version 0.2.1
    PIL version:  4.2.1


## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```


```python
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms_train = transforms.Compose([transforms.RandomRotation(30, False, True),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
data_transforms_valid = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
data_transforms_test = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
# TODO: Load the datasets with ImageFolder
image_dataset_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
image_dataset_valid = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
image_dataset_test  = datasets.ImageFolder(test_dir, transform=data_transforms_test)


# TODO: Using the image datasets and the transforms, define the dataloaders
dataloader_train  = torch.utils.data.DataLoader(image_dataset_train, batch_size=64, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(image_dataset_valid, batch_size=32)
dataloader_test  = torch.utils.data.DataLoader(image_dataset_test, batch_size=32)

# Number of images in the loaders
train_num = len(image_dataset_train.imgs)
print("Train dataset count:     ", train_num)

valid_num = len(image_dataset_valid.imgs)
print("Validation dataset count:", valid_num)

test_num = len(image_dataset_test.imgs)
print("Test dataset count:      ",test_num)

class_to_idx = image_dataset_train.class_to_idx
```

    Train dataset count:      6552
    Validation dataset count: 818
    Test dataset count:       819


### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

## Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.

Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

###  Build the network 


```python
def build_model_vgg19(class_to_idx, 
                      hidden_units=12595, 
                      input_size=25088, 
                      output_size=102, 
                      dropout=0.6):
    
    # Loading pre-trained network to do transfer learning
    model = models.vgg19(pretrained=True)

    # freezing parameter of feature network
    for param in model.parameters():
        param.requires_grad = False
    
    # Define feed a new forward classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('logsoftmax', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    return model
```


```python
hidden_units = 12595
model = build_model_vgg19(class_to_idx, hidden_units)
model
```




    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace)
        (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (17): ReLU(inplace)
        (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace)
        (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (24): ReLU(inplace)
        (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (26): ReLU(inplace)
        (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace)
        (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (31): ReLU(inplace)
        (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (33): ReLU(inplace)
        (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (35): ReLU(inplace)
        (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (fc1): Linear(in_features=25088, out_features=12595, bias=True)
        (relu): ReLU()
        (dropout1): Dropout(p=0.6)
        (fc2): Linear(in_features=12595, out_features=102, bias=True)
        (logsoftmax): LogSoftmax()
      )
    )



### Methods for validation and training


```python

def validate(model, criterion, dataloader, percent=100):
    model.eval()
    accuracy = 0
    valid_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def train(model, criterion, optimizer, loader_train, loader_valid, epochs, learn_rate):
    
    data_len = len(loader_train)
    validate_every = data_len # see validation 1 time by epoch
    steps = 0
    initial_t = time.time()
    
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in iter(loader_train):
            epoch_status = 100*(steps%data_len)/data_len
            print("", end='\r')
            print("Epoch {:d}/{:d} - {:3.0f}%".format(e+1,epochs,epoch_status),end="")
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            
            if steps % validate_every == 0 or steps==1:
                print()
                valid_loss, accuracy = validate(model, criterion, loader_valid)
                print("Valid Loss: {:.3f} ".format(valid_loss),
                      "Valid Accuracy %: {:.3f}".format(accuracy))

                running_loss = 0
                model.train()
```

### Training


```python
learn_rate = 0.003
momentum = 0.9
epochs = 3 # with just 3 epochs gets to > 70% accuracy

if torch.cuda.is_available():
    print("Using GPU")
    print("  Device name:", torch.cuda.get_device_name(torch.cuda.device_count()-1))
    print("  Number of GPUs:", torch.cuda.device_count())
else:
    print("Using CPU")

# Loss function
criterion = nn.NLLLoss()
# Optimizer (Only train the classifier parameters)
optimizer = optim.SGD(
    model.classifier.parameters(),
    lr=learn_rate,
    momentum=momentum
)

print("Starting training:")
train(model, criterion, optimizer, dataloader_train, dataloader_valid, epochs, learn_rate)
```

    Using GPU
      Device name: Tesla K80
      Number of GPUs: 1
    Starting training:
    Epoch 1/3 -   0%
    Valid Loss: 4.626  Valid Accuracy %: 1.442
    Epoch 1/3 -  99%
    Valid Loss: 1.662  Valid Accuracy %: 59.348
    Epoch 2/3 -  99%
    Valid Loss: 1.043  Valid Accuracy %: 71.955
    Epoch 3/3 -  99%
    Valid Loss: 0.707  Valid Accuracy %: 82.025


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
# TODO: Do validation on the test set
valid_loss, accuracy = validate(model, criterion, dataloader_test)
print("Validating in test data        ")
print("  Loss:     {:.1f}".format(valid_loss))
print("  Accuracy: {:.1f}%".format(accuracy))
```

    Validating in test data        y:  77%  
      Loss:     0.8
      Accuracy: 76.5%


## Checkpoint

### Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
def save_checkpoint(filename_path, model, epochs, learn_rate, hidden_units):

    state = {
        'arch': 'vgg19',
        'learn_rate': learn_rate,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx' : model.class_to_idx
    }

    torch.save(state, filename_path)
```


```python
save_checkpoint('vgg19_checkpoint.pth', model, epochs, learn_rate, hidden_units)
```

### Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
def load_checkpoint( filename_path):
    state = torch.load(filename_path)
    
    arch = state['arch']
    learn_rate = state['learn_rate']
    hidden_units = state['hidden_units']
    epochs = state['epochs']
    class_to_idx = state['class_to_idx']

    model = build_model_vgg19(class_to_idx, hidden_units)
    model.load_state_dict(state['state_dict'])
    
    return model, epochs, learn_rate, hidden_units
```


```python
model, epochs, learn_rate, hidden_units = load_checkpoint('vgg19_checkpoint.pth')
model.classifier
```




    Sequential(
      (fc1): Linear(in_features=25088, out_features=12595, bias=True)
      (relu): ReLU()
      (dropout1): Dropout(p=0.6)
      (fc2): Linear(in_features=12595, out_features=102, bias=True)
      (logsoftmax): LogSoftmax()
    )




```python
# Validating the loaded checkpoint model
valid_loss, accuracy = validate(model, criterion, dataloader_test)
print("Validating in test data        ")
print("  Loss:     {:.1f}".format(valid_loss))
print("  Accuracy: {:.1f}%".format(accuracy))
```

    Validating in test data        y:  77%  
      Loss:     0.8
      Accuracy: 76.5%


## Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    image = Image.open(image_path)
    
    resize_size = 256
    final_size = 224
    width, height = image.size
    ratio = float(width/height);
        
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
```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
```


```python

image_path = 'flowers/test/91/image_08061.jpg'
imshow(process_image(image_path))
the_flower= cat_to_name['91']
print("Flower Class: {}".format(the_flower))
```

    Flower Class: hippeastrum



![png](output_31_1.png)


## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()

    np_array = process_image(image_path)
    tensor_in = torch.from_numpy(np_array)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
#     top_labels = [cat_to_name[cat] for cat in top_cats ]

    return topk_probs, top_cats# , top_labels

image_path = 'flowers/test/91/image_08061.jpg'
probabilities, classes = predict(image_path, model)

print(probabilities)
print(classes)
```

    [0.9965574145317078, 0.0025194825138896704, 0.00034203173709101975, 0.00025675276992842555, 5.6185443099820986e-05]
    ['91', '85', '18', '31', '40']


## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='https://github.com/udacity/aipnd-project/blob/master/assets/inference_example.png?raw=true' width=300px >

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
# TODO: Display an image along with the top 5 classes

image_path = 'flowers/valid/28/image_05265.jpg'

probs, classes = predict(image_path, model)

top_labels = [cat_to_name[cat] for cat in classes ]
print('probabilities:', probs, '\nclasses:', top_labels, '\n')

max_index = np.argmax(probs)
max_probability = probs[max_index]
label = classes[max_index]

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
```

    probabilities: [0.9607481956481934, 0.01938352733850479, 0.00582913588732481, 0.0034667393192648888, 0.0032432549633085728] 
    classes: ['stemless gentian', 'bolero deep blue', 'canterbury bells', 'morning glory', 'spring crocus'] 
    



![png](output_35_1.png)

