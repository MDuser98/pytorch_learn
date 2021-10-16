import torch
import math

from torch.nn.modules import linear

#create tensors to hold input and outputs
x=torch.linspace(-math.pi,math.pi,2000)
y=torch.sin(x)

#for this examples, the output y is a linear function of (x,x^2,x^3), so we can consider it as a linear layer neural network 
#let us prepare the tensor (x,x^2,x^3)
p=torch.tensor([1,2,3])
xx=x.unsqueeze(-1).pow(p)

#in the above code, x.unspueeze(-1) has shape (2000,1), and p has shape (3,), for this case, broadcasting semantics will apply to obtain a tensor of shape (2000,3)

#use the nn package to define our model as a sequence of layers.
#nn.Squential is a Module which contains other Modules, and applies them in sequence to produce its output.
#the linear Module computes output from input using a linear function, and holds internal tensors for its weight and bias.
# the flatten layer flatens the output of the linear layer to a 1D tensor, to match the shape of 'y'
model = torch.nn.Sequential(
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)
)                                                                       

# the nn package also contains definitions of popular loss functions; in this case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):

    #forward pass: compute predicted y by passing x to the model. Module objects override the __call__ operator 
    #so you can call them like functions.
    #when doing so you pass a tensor of input data to the Module and it produces a tensor of output data
    y_pred = model(xx)

    # compute and print loss.
    # we pass tensors containing the predicted and True values of y, and the loss function returns a tensor containing the loss
    loss = loss_fn(y_pred,y)
    if t % 100 == 99:
        print(t,loss.item())

    #zero the gradients before running the backward pass.
    model.zero_grad()

    #backward pass: compute gradient of the loss with respect to all the learnable parameters of the model.
    # Internally, the parameters of each Module are stored in tensors with requires_grad=True, so this call 
    # will compute gradients for all learnable parameters in the model.
    loss.backward()

    # update the weights using gradient descent.
    # each parameter is a tensor, so we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

#you can access the first layer of 'model' like accessing the first item of a list 
linear_layer = model[0]

#For linear layer, its parameters are stored as 'weight' and 'bias'.
print(f'Results: \n  y={linear_layer.bias.item()} + {linear_layer.weight[:,0].item()}x + {linear_layer.weight[:,1].item()}x^2 + {linear_layer.weight[:,2].item()}x^3 \n')

