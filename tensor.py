import torch
import math

dtype=torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")  #uncomment this to run on gpu

#create random input and output data
x = torch.linspace(-math.pi,math.pi,2000,device=device,dtype=dtype)
y=torch.sin(x)

#random initialize weights
a = torch.randn((),device=device,dtype=dtype)
b = torch.randn((),device=device,dtype=dtype)
c = torch.randn((),device=device,dtype=dtype)
d = torch.randn((),device=device,dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    #forward pass: compute predicted y
    y_pred = a + b*x + c*x**2 + d*x**3

    loss = (y_pred - y).pow(2).sum().item()
    if t%100 == 99:
        print(t,loss)

    # backprop to compute gradients of a,b,c,d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    #update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Rsults: \n y={a}+{b}x+{c}x^2+{d}x^3\n')
