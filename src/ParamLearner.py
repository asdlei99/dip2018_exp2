#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class ParamLearner(models.AlexNet):
    def __init__(self, hidden_size=4096):
        super(ParamLearner, self).__init__() 
        self.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        for param in self.parameters():
            param.requires_grad = False

        # Delete last fc layer
        self.classifier.__delitem__(6)

        # Define param learner
        self.param_learner = nn.Linear(hidden_size, hidden_size)

        # Initialized with identity matrix
        self.param_learner.weight.data.copy_(torch.eye(hidden_size))  
    
    def forward(self, x, R):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)

        # R is a (num_class, hidden_size) matrix, w is a (num_class, hidden_size) matrix
        w = self.param_learner(R) 
        x = torch.matmul(x, w.transpose(0, 1))
        return x
    
    def get_r(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def forward_test(self, x, Rs):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        logits = [] 
        for i in range(x.shape[0]):
            i_logits = []
            for class_Rs in Rs:
                # class_Rs is a (n, hidden_size) matrix, in which n is the number of training pictures of this class.
                class_w = self.param_learner(class_Rs) 
                class_logit = torch.matmul(class_w, x[i])
                i_logits.append(class_logit.max())
            logits.append(torch.stack(i_logits))
        return torch.stack(logits)

