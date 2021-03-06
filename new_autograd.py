import torch
import math

class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing torch.autograd.Function and implementing the forward and backward passes which operates on Tensors.
    """

    @staticmethod
    def forward(ctx,input):
        """
        In the forward pass we receive a tensors containing the input and return a tensors containing the output.
        ctx is a context object that can be used to stash information for backward computation.
        you can cache arbitrary objects for use in the backward pass using the ctx.save_for_backward mathod
        """
        ctx.save_for_backward(input)
        return 0.5 * (5*input**3-3*input)

    @staticmethod
    def backward(ctx,grad_output):
        """
        In the backward pass we receive a tensor containing the gradient of the loss with respect to the output,
        and we need to compute the gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input **2 -1)

dtype = torch.float 
device = torch.device("cpu")
# device = torch.device("cuda:0") # uncomment this to gpu

# create tensors to hold input and outputs
# by default, requires_grad=False, which indicates that we do not need to compute gradients with respect to these tensors during the backward pass.
x= torch.linspace(-math.pi,math.pi,2000,device=device,dtype=dtype)
y=torch.sin(x)

# Create random tensors for weights. For this examples, we need 4 weights: y = a + b * P3 (c + d * x), 
# these weights need to be initialized not too far from the correct result to ensure convergence.
# compute gradients with respect to these tensors during the backward pass
a= torch.full((),0.0,device=device,dtype=dtype,requires_grad=True)
b= torch.full((),-1.0,device=device,dtype=dtype,requires_grad=True)
c= torch.full((),0.0,device=device,dtype=dtype,requires_grad=True)
d= torch.full((),0.3,device=device,dtype=dtype,requires_grad=True)

learning_rate=5e-6
for t in range(2000):
    #to apply our function, we use function.apply method. We alias the as P3
    P3 = LegendrePolynomial3.apply

    #Forward pass: compute predicted y using operations; we compute P3 using our custom autograd operation.
    y_pred = a + b*P3(c+d*x)

    #compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t,loss.item())

        #use autograd to compute the backward pass.
        loss.backward()

        #update weights using gradient descent
        with torch.no_grad():
            a -= learning_rate*a.grad
            b -= learning_rate*b.grad
            c -= learning_rate*c.grad
            d -= learning_rate*d.grad

            #manually zero the gradients after uspdating weights
            a.grad=None
            b.grad=None
            c.grad=None
            d.grad=None

print(f"Resultes: \n y= {a.item()}+{b.item()}*P3({c.item()}+{d.item()}x) \n")
