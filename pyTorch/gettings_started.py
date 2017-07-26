#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
from __future__ import print_function
import torch
import numpy as np

########################
# Tensors
########################

# Construct a 5x3 matrix, uninitialized
x = torch.Tensor(5, 3)

# Construct a randomly initialized matrix
x = torch.rand(5, 3)
print(u"Construct a randomly initialized matrix: ")
print(x)

# Get its size
print(u"Get its size: ")
print(x.size())

########################
# Operations
########################

# Addition: syntax 1
y = torch.rand(5, 3)
print(u"Addition: syntax 1")
print(x + y)

# Addition: syntax 2
print(u"Addition: syntax 2")
print(torch.add(x, y))

# Addition: giving an output tensor
print(u"Addition: giving an output tensor")
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition: in-place
y.add_(x)
print(u"Addition: in-place")

# First column
print(u"First column: ")
print(x[:, 1])

########################
# Tensor to numpy array
########################

a = torch.ones(5)
print(u"One tensor of size 5")
print(a)

# Tensor to Numpy Array
b = a.numpy()
print(u"Tensor to numpy array")
print(b)

# Change array
a.add_(1)
print(u"See how the numpy array changed in value")
print(a)
print(b)

########################
# Numpy array to Tensor
########################

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(u"Numpy array to Tensor")
print(u"Tensor")
print(a)
print(u"Numpy array")
print(b)

########################
# CUDA Tensors
########################

# Let us run this cell only if CUDA is available
if torch.cuda.is_available():
    print(u"CUDA is available")
    x = x.cuda()
    y = y.cuda()
    print(x + y)
# end if
