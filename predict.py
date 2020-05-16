# PROGRAMMER: Ajay Sukumar
# DATE CREATED:  15/05/2020
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np 
import pandas as pd
import collections
import json
from time import time

def get_input_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('image_path', help = 'Path to the image to test', type = str)
        parser.add_argument('checkpoint', help = 'path along with checkpoint filename (eg: checkpoint.pth)', type =str)
        parser.add_argument('--top_k' , help = 'Top K most likely classes', default = '5', type = int)
        parser.add_argument('--category_names', help = 'path to json mapping of categories to flower names', default = './cat_to_name.json', type = str)
        parser.add_argument('--gpu', type = str, help = 'use GPU (yes/no)', default = 'yes')
        
        return parser.parse_args()
        
def load_checkpoint(args):
    checkp = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)
    arch = checkp['arch']
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    
    model.classifier = checkp['Classifier']
    model.load_state_dict(checkp['state_dict'])
    model.class_to_idx = checkp['class_to_idx']
    return model

def load_cat_to_names(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    cat_to_name = collections.OrderedDict(sorted(cat_to_name.items()))
    return cat_to_name

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    width,height = image.size
    if width == height:
        size = 256, 256
    elif width > height:
        size = 256* (width/height), 256
    elif height > width:
        size = 256, 256*(height/width)
        
    image.thumbnail(size, Image.ANTIALIAS)
    width, height  = image.size
    left = (width - 224)/2
    upper = (height - 224)/2
    right = (width + 224)/2
    lower = (height + 224)/2
    image= image.crop((left, upper, right, lower))

    np_image = np.array(image)/np.max(image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = ((np_image - mean) / std)
    
    np_image = np_image.transpose(2, 0, 1)
    tensor_image = torch.FloatTensor(np_image)
    return tensor_image

def predict(args,model):
    cat_to_names = load_cat_to_names(args.category_names)
    image_path = args.image_path
    cuda = torch.cuda.is_available()
    if cuda and args.gpu == 'yes':
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model.eval()
    with torch.no_grad():
        im = process_image(image_path)
        im_tensor = im.unsqueeze(0)
        im_tensor = im_tensor.float().to(device)
        output = model(im_tensor)
        ps=torch.exp(output)
        probs,classes= ps.topk(args.top_k,dim = 1)    
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        prob_arr = probs.data.cpu().numpy()[0].tolist()
        pred_indexes = classes.data.cpu().numpy()[0].tolist()    
        pred_labels = [idx_to_class[x] for x in pred_indexes]
        pred_class = [cat_to_names[str(x)] for x in pred_labels]
    model.train()
    return prob_arr,pred_class

def show_prediction(probs, f_names):
    for i in range(len(probs)):
        print("Probability that image is of type {} is {:.3f}.".format(
            f_names[i], probs[i]))

def main():
    start_time = time()
    in_arg = get_input_args()
    print("\nLoading the trained model...\n")
    model = load_checkpoint(in_arg)
    print("\nSuccessfully loaded")
    print("\nFinding the class of input image...") 
    probs, f_names = predict(in_arg,model)
    show_prediction(probs,f_names)
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Runtime:",
          str(int((tot_time / 3600))) + ":" + str(
              int((tot_time % 3600) / 60)) + ":"
          + str(int((tot_time % 3600) % 60)))
    print("DONE...\n\n")
    
    
if __name__ == '__main__':
    main()