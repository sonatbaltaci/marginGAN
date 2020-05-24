import torch.optim as optim
import torchvision.transforms as transforms
import argparse
import time
import datetime
import os
import tqdm
import numpy as np
from marginGAN import *
from utils import *
from dataset import *

torch.manual_seed(0)
batch_size = 50
num_epochs = 10000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = torchvision.datasets.MNIST("./data",train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]),target_transform=None,download=True)
trainset, _ = torch.utils.data.random_split(dataset, [50000, 10000])
testset = torchvision.datasets.MNIST("./data",train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]),target_transform=None,download=True)


testloader = torch.utils.data.DataLoader(
					dataset=testset,
					batch_size=batch_size,
					shuffle=False)

for label_size in [100,600,1000,3000]:
    labeled_set, _ = divide_dataset(trainset, label_size)
    trainloader = torch.utils.data.DataLoader(dataset=labeled_set,batch_size=batch_size,shuffle=True)
    C = Classifier()
    C = C.to(device)
    optimizer = optim.SGD(C.parameters(),lr=0.1,momentum=0.8)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        C.train()
        for i,(image,label) in enumerate(trainloader,0):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs, _ = C(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        correct = 0
        total = 0
        C.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = C(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            if label_size == 100 and correct/total > 0.80:
                break
            elif label_size == 600 and correct/total > 0.93:
                break
            elif label_size == 1000 and correct/total > 0.95:
                break
            elif label_size == 3000 and correct/total > 0.97:
                break

        print('[',epoch+1,']','Accuracy: %d %%' % (
                100 * correct / total), correct,"/", total)
        
    save_path = os.path.join("pretrained_classifiers","pre_cls_label_model_"+str(label_size)+".pt")
    torch.save(C, save_path)
    
    
