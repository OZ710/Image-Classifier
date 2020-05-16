# PROGRAMMER: Ajay Sukumar
# DATE CREATED:  14/05/2020
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np 
import pandas as pd
import collections
import json
from time import time
import os

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',type = str, help = 'path to data directory containing train,validation and test data')
    parser.add_argument('--save_dir',type = str, help = 'directory to save the trained model')
    parser.add_argument('--arch', type = str, help = 'CNN architecture', default = 'vgg', choices = ['vgg', 'densenet'])
    parser.add_argument('--learning_rate',type = float, help = 'learning rate for the optimizer', default = '0.001')
    parser.add_argument('--epochs', type = int, help = 'number of epochs to train the model', default = '10')
    parser.add_argument('--hidden_units', type = int, help = 'number of hidden units for the model', default = '512')
    parser.add_argument('--gpu', type = str, help = 'use GPU (yes/no)', default = 'yes')
    return parser.parse_args()

def check_args(args):
    print("Checking command line arguments...")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...GPU not found")
    if(not os.path.isdir(args.data_dir)):
        raise Exception('Inavlid directory')
    data_dir = os.listdir(args.data_dir)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('missing: test, train or valid sub-directories')
    if args.arch not in ('vgg','densenet',None):
        raise Exception('invalid Architecture, choose either vgg or densenet')
        
def load_data(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir,transform = test_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,batch_size = 64,shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size = 32)
    testloader = torch.utils.data.DataLoader(test_data,batch_size = 32)
    dataloader = {'train' : trainloader, 'valid' : validloader, 'test' : testloader}
    return dataloader, train_data

def build_model(args):
    arch_type = args.arch
    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        input_size=25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_size=1024
        
    for param in model.parameters():
        param.requires_grad = False
    hidden_size = args.hidden_units
    learn_rate = args.learning_rate
    if hidden_size >= input_size:
        raise ValueError("hidden units must be lower than input size")

    if hidden_size <= 102:
        raise ValueError("hidden units must be greater than output size")

    classifier = nn.Sequential(nn.Linear(input_size,hidden_size),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_size,102),
                               nn.LogSoftmax(dim = 1)
                              )
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    return model,criterion,optimizer
                         
def train_model(model, dataloader, criterion, optimizer, args):
    cuda = torch.cuda.is_available()
    if cuda and args.gpu == 'yes':
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    epochs = args.epochs
    step = 0
    running_loss = 0
    print_every = 60
    trainloader = dataloader['train']
    validloader = dataloader['valid']
    for epoch in range(epochs):
        for images,labels in trainloader:
            step+=1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.forward(images)
            loss = criterion(logits,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

            if step % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logits = model.forward(images)
                        loss = criterion(logits,labels)
                        test_loss += loss.item()

                        ps = torch.exp(logits)
                        top_p ,top_class = ps.topk(1,dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch + 1}/{epochs}..."
                      f"Train loss : {running_loss/print_every:.3f}..."
                      f"Test loss : {test_loss/len(validloader):.3f}..."
                      f"Accuracy : {accuracy/len(validloader):.3f}...") 
                running_loss = 0
                model.train()
    print("\nModel Trained\n")
    return model

def test_model(model,args,dataloader,criterion):
    cuda = torch.cuda.is_available()
    if cuda and args.gpu == 'yes':
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    test_loss = 0
    accuracy = 0
    model.eval()
    testloader = dataloader['test']
    with torch.no_grad():
        for images, labels in testloader:
            images , labels = images.to(device), labels.to(device)
            logits = model.forward(images)
            loss = criterion(logits,labels)
            test_loss += loss.item()
            ps = torch.exp(logits)
            top_p ,top_class = ps.topk(1,dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"Test loss : {test_loss/len(testloader):.3f}..."
              f"Test Accuracy : {accuracy/len(testloader):.3f}...") 
        
def save_model(args,model,optimizer):
    if (args.arch is None):
        arch_type = 'vgg19'
    else:
        if (args.arch == 'vgg'):
            arch = 'vgg19'
            input_size=25088
        elif (args.arch == 'densenet'):
            arch = 'densenet121'
            input_size=1024
      
    
    if (args.save_dir is None):
        checkpoint = 'trained_model.pth'
    else:
        checkpoint = args.save_dir + '/' + 'trained_model.pth'
    model_info = {'Classifier' : model.classifier,
              'arch' : arch,
              'state_dict' : model.state_dict(),
              'epochs' : args.epochs,
              'optimizer_state' : optimizer.state_dict(),
              'class_to_idx' : model.class_to_idx,
              'input_size' : input_size,
              'output_size' : 102}
    torch.save(model_info,checkpoint)

def main():
    start_time = time()
    in_arg = get_input_args()
    check_args(in_arg)
    print("Loading the data...\n")
    dataloader, train_data = load_data(in_arg)
    print("Building the model\n")
    model,criterion,optimizer = build_model(in_arg)
    print("Model successfully built\n")
    print("Training the model...\n")
    trained_model = train_model(model,dataloader,criterion,optimizer,in_arg)
    test_model(model,in_arg,dataloader,criterion)
    model.class_to_idx = train_data.class_to_idx
    save_model(in_arg,model,optimizer)
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Runtime:",
          str(int((tot_time / 3600))) + ":" + str(
              int((tot_time % 3600) / 60)) + ":"
          + str(int((tot_time % 3600) % 60)))
    print("DONE...\n\n")
if __name__ == '__main__':
    main()
                         
    
    