#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# load and normalize the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     #y = (x-0.5)/0.5 --> x = (y * 0.5) — 0.5 = (y / 2) + 0.5.
     ]
    )

batch_size = 4
num_workers = 2 #perform multi-process data loading
trainset = torchvision.datasets.CIFAR10(root='./data',train = True,transform=transform,download=True)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,num_workers=num_workers)
testset = torchvision.datasets.CIFAR10(root='./data',train = False,transform=transform,download=True)
testloader = torch.utils.data.DataLoader(testset,batch_sampler=batch_size,num_workers=num_workers)
classes = ('plane','car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck') #tuple

'''
def imshow(images):
    images = images/2+0.5
    npimg = images.numpy()
    #plt.imshow(num_image.permute(1, 2, 0)) # permute for tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #tranpose for numpy,change the position of dimension
    plt.show()

# show a image
dataiter = iter(trainloader)
images,labels = dataiter.next()
grid_img = torchvision.utils.make_grid(images)
imshow(grid_img)
# print the class of the image
print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))
'''

# define a network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        ## in_features由输入张量的形状决定，out_features则决定了输出张量的形状 
        # every neuron from the previous layers connects to all neurons in the next
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self,x):
        x = self.pool1(F.relu((self.conv1(x))))
        x = self.pool2(F.relu(self.conv2(x)))
        # we do not know how many rows,so we set -1 to auto calculate
        # we can also use 'x.flatten(1)'
        x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

# define a loss function
# CrossEntropyLoss combines softmax and negative log-likelihood
# softmax: scales numbers into probabilities for each outcome. These probabilities sum to 1.
# negative log-likelihood： small value ——> hegh loss
criterion = nn.CrossEntropyLoss()
#  net.parameters():  gets the learnable parameters 
#  momentum: faster converging
# lr: how big the step
optimizer = optim.SGD(net.parameters(), lr=0.001,momentum = 0.9)

# start train the network
# time the training
start = torch.cuda.Event(enable_timing = True)
end = torch.cuda.Event(enable_timing = True)

start.record()

for epoch in range(2):
    running_loss = 0.0
    # (trainloader,0) start from index = 0
    # (i, data) index and content
    for i, data in enumerate(trainloader,0):
        # trainloader  [inputs,labels]
        inputs,labels = data # data is an array
        # zero the parameter gradients
        optimizer.zero_grad()
        #forward
        outputs = net(inputs)
        # backward + optimize
        loss = criterion(outputs,labels)
        loss.backward() # backward is PyTorch’s way to perform backpropagation by computing the gradient based on the loss
        optimizer.step() 
        # item():one element,tensor to numpy
        # loss:(1,) tensorVariable
        running_loss +=loss.item()
        # print loss every mini-batch
        if i%2000 == 1999:
            print('[%d %5d] loss: %.3f' %[epoch+1, i+1,running_loss/2000])
            running_loss = 0.0
            
end.record()        
# Waits for everything to finish running
torch.cuda.synchronize()
print('Finish training') 
print('Training time: %.3f' %start.elapsed_time(end)) #ms    
        
# save the network   
path = './simple_cnn.pth'
torch.save(net.state_dict(),path)
#reload
#net = Net()
#net.load_state_dict(torch.load(path))

#test the network on testing data
dataiter = iter(testloader)
images,labels = dataiter.next()
outputs = net(images)
_,predicts = torch.max(outputs,1) # value,index; 1 for row
print('Prediction:',''.join('%s'%classes[predicts(j)] for j in range(4)))

# test on 10000 images
correct = 0 
total = 0
with torch.no_grad(): #do not calculate the gradient
    for (i,data) in enumerate(testloader):
        test_images,labels = data
        output = net(test_images)
        #_,prediction = torch.max(output,1)
        _,prediction = torch.max(output.data,1)
        total += labels.size(0)
        # (prediction==labels) bool tensor array
        # (prediction==labels).sum() : eg. tensor(3)
        correct += (prediction==labels).sum().item()
print('the testing accuracy on 10000 images: %d %%' %(100*correct/total))            