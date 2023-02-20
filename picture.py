#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:39:10 2023

@author: liyang
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import matplotlib.pyplot as plt

x=torch.arange(-8,8)
relu=nn.ReLU()

y = relu(x.float())

plt.plot(x,y)

plt.show()