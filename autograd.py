import torch
import math

dtype = torch.float
device = torch.device('cpu')

# create tensor to hold input and outputs
# by  default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these tensors during the backward pass
x = torch.linspace(-math.pi,math.pi, 2000, device=device,dtype=dtype)
y=torch.sin(x)

# create random tensors for weights. for a third order polynomial, we need 4 weights: y=a+bx+cx**2+dx**3
# setting requires_grad=True indicates that we want to compute gradients with respect to these tensors during the backward pass
a = torch.randn((),device=device,dtype=dtype,requires_grad=True)
b = torch.randn((),device=device,dtype=dtype,requires_grad=True)
c = torch.randn((),device=device,dtype=dtype,requires_grad=True)
d = torch.randn((),device=device,dtype=dtype,requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    #forward pass: compute predicted y using operations on tensors
    y_pred = a+b*x+c*x**2+d*x**3

    # compute and print loss using operations on tensors
    # now loss is a tensors of shape(1)
    # loss.item() gets the scalar value held in the loss
    loss = (y_pred-y).pow(2).sum()
    if t% 100 == 99:
        print(t,loss.item())
    
    # use autograd to compute the backward pass. This call will compute the gradient of loss with reapect to all tensors with requires_grad=True.
    # After this call a.grad, b.grad, c.grad, d.grad will be tensors holding the gradient of the loss with respect to a, b, c, d respectively
    loss.backward()
#    grad_y_pred = 2.0 * (y_pred - y)
#    grad_a = grad_y_pred.sum()
#    grad_b = (grad_y_pred*x).sum()
#    grad_c = (grad_y_pred*x**2).sum()
#    grad_d = (grad_y_pred*x**3).sum()

    #manually update weights using gradient descent. Wrap in torch.no_grad() because weights have requires_grad=True, but we don't need to track this in autograd
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"Results : \n  y= {a.item()}+{b.item()}x+{c.item()}x^2+{d.item()}x^3  \n")

    