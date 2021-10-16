import torch
import math

from torch.nn.modules import linear

#create tensors to hold input and outputs
x = torch.linspace(-math.pi,math.pi,2000)
y=torch.sin(x)

#prepare the input tensor (x,x^2,x^3)
p=torch.tensor([1,2,3])
xx=x.unsqueeze(-1).pow(p)

#use the nn package to define our model and loss function
model = torch.nn.Sequential(
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# use the optim package to define an optimizer that will update the weights of the model for us.
# Here we will use RMSprop; the optim package contains many other optimization algorithms.
# the first argument to the RMSprop constructor tells the optimizer which tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(),lr=learning_rate)
for t in range(2000):
    #forward pass: compute predicted y by passing x to the model.
    y_pred = model(xx)

    #compute and print loss.
    loss = loss_fn(y_pred,y)
    if t % 100 == 99:
        print(t,loss.item())

    # before the backward pass, use the optimizer object to zero all of the gradients for the variable it will update(which are the learnable weights of the model).
    # this is because by default, gradients are accumulated in buffers( i.e, not overwritten) whenever .backward() is called.
    # checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    #backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    #calling the step function on an optimizer makes an update to its parameters
    optimizer.step()

linear_layer = model[0]

print(f'Results: \n  y={linear_layer.bias.item()} + {linear_layer.weight[:,0].item()} x + {linear_layer.weight[:,1].item()} x^2 + {linear_layer.weight[:,2].item()} x^3 \n')