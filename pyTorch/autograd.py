#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
from __future__ import print_function
import torch
from torch.autograd import Variable

# Create a variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print(u"Variable x: ")
print(x)

# Do an operation of variable
print(u"y = x + 2")
y = x + 2
print(y)
print(u"y.creator")
print(y.creator)

# More operations on y
z = y * y * 3
out = z.mean()
print(u"z, out")
print(z, out)

########################
# Gradients
########################

# Let's backprop
out.backward()

# Print gradient
print(u"gradients d(out)/dx")
print(x.grad)

########################
# Crazy things with Autograd
########################

x = torch.randn(1)
print(u"X: ")
print(x)
x = Variable(x, requires_grad=True)

y = x * 2
index = 1
while y.data.norm() < 1000:
    y = y * 2
    index += 1
# end while
print(u"How many passes: ")
print(index)

print(u"Result y: ")
print(y)
print(y.creator)

# Forward pass

# Get gradients
print(u"Backward pass: ")
gradients = torch.FloatTensor([0.001, 0.01, 0.1])
y.backward(torch.FloatTensor([1.0]))
print(u"Gradients: ")
print(x.grad)
print(x.data)

# Get gradients
#y.backward()
#print(u"Gradients: ")
#print(x.grad)
