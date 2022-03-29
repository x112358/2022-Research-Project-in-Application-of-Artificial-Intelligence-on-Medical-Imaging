import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import random
import cv2
import copy
import os
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
def train(model, criterion, optimizer, scheduler, epochs):
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    f1_score = []
    store_test_true = []
    store_test_pred = []
    for epoch in range(epochs):
        
        for phase in ['train', 'test']:
            TP=0
            TN=0
            FP=0
            FN=0
            if phase == 'train':
                model.train()
            else:
                model.eval()
                    
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outp = model(inputs)
                    _, pred = torch.max(outp, 1)
                    loss = criterion(outp, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()*inputs.size(0)
                #running_corrects += torch.cpu().sum(pred == labels.data)
                running_corrects += (pred == labels.data).cpu().sum().numpy()
                
                if phase == 'test':
                    
                    for j in range(inputs.size()[0]):
                        if epoch == epochs-1:
                            store_test_true.append(int(labels[j]))
                            store_test_pred.append(int(pred[j]))
                        if (int (pred[j]) == 1 and int (labels[j]) ==  1):
                            TP += 1
                        if (int (pred[j]) == 0 and int (labels[j]) ==  0):
                            TN += 1
                        if (int (pred[j]) == 1 and int (labels[j]) ==  0):
                            FP += 1
                        if (int (pred[j]) == 0 and int (labels[j]) ==  1):
                            FN += 1
            if phase == 'test':
                Recall = TP/(TP+FN)
                print ("Recall : " ,  Recall )
                
                Precision = TP/(TP+FP)
                print ("Precision : " ,  Precision )
                
                F1_score = 2 * Precision * Recall / (Precision + Recall)
                print ("F1 - score : " , F1_score)
                f1_score.append(F1_score)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects/ dataset_sizes[phase]
            
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            if phase == 'train':
                print('Epoch: {}/{}'.format(epoch+1, epochs))
                print('{} - loss:{},  accuracy{}'.format(phase, epoch_loss, epoch_acc))



            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
            scheduler.step()  
  
    print('Best accuracy {}'.format(best_acc))

    model.load_state_dict(best_model)
    return model,f1_score,store_test_true,store_test_pred,accuracies

if __name__=="__main__":
    path = 'chest_xray/chest_xray'
    transformer = {
    'dataset':transform.Compose([
            transform.Resize(255),
            transform.CenterCrop(224),
            transform.RandomHorizontalFlip(),
            transform.RandomRotation(10),
            transform.RandomGrayscale(),
            transform.RandomAffine(translate=(0.05,0.05), degrees=0),
            transform.ToTensor()
        ]),
              }
    train_dataset =  ImageFolder(path+'/train', transform=transformer['dataset'])

    random_seed = 777
    torch.manual_seed(random_seed);
    trainset, testset = train_test_split(train_dataset, test_size=0.3,random_state=777)

    batch_size = 10
    train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size*2, num_workers=4, pin_memory=True)
    loaders = {'train':train_loader, 'test':test_loader}
    dataset_sizes = {'train':len(trainset), 'test':len(testset)}

    model = torchvision.models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.classifier.in_features

    model.classifier = nn.Linear(in_features, 2)
    losses = {'train':[], 'test':[]}
    accuracies = {'train':[], 'test':[]}
    f1_score = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    model.to(device)
    grad_clip = None
    weight_decay = 1e-4
    epochs = 10
    model,f1_score ,store_test_true,store_test_pred , accuracies= train(model, criterion, optimizer, scheduler, epochs)

    train_accurate = []
    test_accurate = []

    for i in range(epochs):
        train_accurate.append(accuracies['train'][i].item())
        test_accurate.append(accuracies['test'][i].item())

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    t = f.suptitle('Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)


    epoch_list = list(range(1,epochs+1))

    ax1.plot(epoch_list, train_accurate)
    ax1.set_xticks(np.arange(0, epochs+1, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Train Accuracy')

    ax2.plot(epoch_list, test_accurate)
    ax2.set_xticks(np.arange(0, epochs+1, 5))
    ax2.set_ylabel('Accuracy Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Test Accuracy')
    
    ax3.plot(epoch_list, f1_score )
    ax3.set_xticks(np.arange(0, epochs+1, 5))
    ax3.set_ylabel('F1_score')
    ax3.set_xlabel('Epoch')
    ax3.set_title('Test_F1_score')
    
    cm  = confusion_matrix(store_test_true, store_test_pred)
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8),cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.xlabel('Predicted Label',fontsize=18)
    plt.ylabel('True Label',fontsize=18)
    plt.show()