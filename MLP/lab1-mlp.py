
# coding: utf-8

# In[ ]:


#run_line_magic('matplotlib', 'inline')


# Lab1 - Multilayer Perceptrons
# ----
# 
# In this lab, we are going through 3 examples of MLP, which covers the implementation from scratch and the standard library.
# 
# - Use `numpy` for feed-forward and gradient computing
# - Use PyTorch **tensor** for feed-forward and automatic differentiation
# - Use PyTorch built-in layers and optimizers
# 
# Before you get started, please install `numpy`, `torch` and `torchvision` in advance.
# 
# We suggest you run the following cells and study the internal mechanism of the neural networks. Moreover, it is also highly recommended that you should tune the hyper-parameters to gain better results.
# 
# Some insights of **dropout** and **xavier initialization** has been adapted from [Mu Li](http://www.cs.cmu.edu/~muli/)'s course [Dive into Deep Learning](http://d2l.ai/index.html).

# ## Dataset and DataLoader
# 
# First of all, we utilize the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for example.
# For simplicity, we use the premade dataset powered by `torchvision`, therefore we don't have to worry about data preprocessing : )
# 
# Before moving on, please check the basic concepts of [Dataset and DataLoader of PyTorch](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

# In[ ]:


import numpy as np
import torch
import torchvision

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=256, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=256, shuffle=True)


# 
# Warm-up: numpy
# --------------
# 
# A fully-connected ReLU network with one hidden layer and no biases, trained to
# predict y from x using **cross-entropy loss**.
# 
# This implementation uses numpy to manually compute the forward pass, loss, and
# backward pass.
# 
# A numpy array is a generic n-dimensional array; it does not know anything about
# deep learning or gradients or computational graphs, and is just a way to perform
# generic numeric computations.
# 
# 

# In[ ]:


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)


# In[ ]:


def cross_entropy(y_pred, y, epsilon=1e-12):
    """
    y_pred is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
        Note that y is **not** one-hot encoded vector. 
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    n = y.shape[0]
    p = softmax(y_pred)
    
    # avoid computing log(0)
    p = np.clip(p, epsilon, 1.)
    
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[np.arange(n), y])
    loss = np.sum(log_likelihood) / n
    return loss


# Calculating gradients manually is prone to error; be careful when doing it yourself.
# If you found it difficult, please refer to these sites([link1](https://madalinabuzau.github.io/2016/11/29/gradient-descent-on-a-softmax-cross-entropy-cost-function.html), [link2](https://deepnotes.io/softmax-crossentropy)).

# In[ ]:


def grad_cross_entropy(y_pred, y):
    """
    y_pred is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector. 
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    n = y.shape[0]
    grad = softmax(y_pred)

    grad[np.arange(n), y] -= 1
    grad = grad / n
    return grad


# In[ ]:


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 256, 784, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

n_epochs = 10
learning_rate = 1e-3
display_freq = 50

for t in range(n_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        # Forward pass: compute predicted y
        x = x.view(x.shape[0], -1)
        x, y = x.numpy(), y.numpy()
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = cross_entropy(y_pred, y)
        if batch_idx % display_freq == 0:
            print('epoch = {}\tbatch_idx = {}\tloss = {}'.format(t, batch_idx, loss))

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = grad_cross_entropy(y_pred, y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


# 
# PyTorch: Tensors and autograd
# -------------------------------
# 
# A fully-connected ReLU network with one hidden layer and no biases, trained to
# predict y from x by minimizing cross-entropy loss.
# 
# This implementation computes the forward pass using operations on PyTorch
# Tensors, and uses PyTorch autograd to compute gradients.
# 
# 
# A PyTorch Tensor represents a node in a computational graph. If ``x`` is a
# Tensor that has ``x.requires_grad=True`` then ``x.grad`` is another Tensor
# holding the gradient of ``x`` with respect to some scalar value.
# 
# 

# ### Activation Function

# In[ ]:


def activation(x, method='relu'):
    assert method in ['relu', 'sigmoid', 'tanh'], "Invalid activation function!"

    if method is 'relu':
        return torch.max(x, torch.zeros_like(x))
    elif method is 'sigmoid':
        return 1. / (1. + torch.exp(-x.float()))
    else:
        pos = torch.exp(x.float())
        neg = torch.exp(-x.float())
        return (pos - neg) / (pos + neg)


# ### Dropout
# #### Robustness through Perturbations
# 
# Let's think briefly about what we expect from a good statistical model. Obviously we want it to do well on unseen test data. One way we can accomplish this is by asking for what amounts to a 'simple' model. Simplicity can come in the form of a small number of dimensions, which is what we did when discussing fitting a function with monomial basis functions. Simplicity can also come in the form of a small norm for the basis funtions. This is what led to weight decay and $\ell_2$ regularization. Yet a third way to impose some notion of simplicity is that the function should be robust under modest changes in the input. For instance, when we classify images, we would expect that alterations of a few pixels are mostly harmless.
# 
# In fact, this notion was formalized by Bishop in 1995, when he proved that [Training with Input Noise is Equivalent to Tikhonov Regularization](https://www.mitpressjournals.org/doi/10.1162/neco.1995.7.1.108). That is, he connected the notion of having a smooth (and thus simple) function with one that is resilient to perturbations in the input. Fast forward to 2014. Given the complexity of deep networks with many layers, enforcing smoothness just on the input misses out on what is happening in subsequent layers. The ingenious idea of [Srivastava et al., 2014](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) was to apply Bishop's idea to the *internal* layers of the network, too, namely to inject noise into the computational path of the network while it's training.
# 
# A key challenge in this context is how to add noise without introducing undue bias. In terms of inputs $\mathbf{x}$, this is relatively easy to accomplish: simply add some noise $\epsilon \sim \mathcal{N}(0,\sigma^2)$ to it and use this data during training via $\mathbf{x}' = \mathbf{x} + \epsilon$. A key property is that in expectation $\mathbf{E}[\mathbf{x}'] = \mathbf{x}$. For intermediate layers, though, this might not be quite so desirable since the scale of the noise might not be appropriate. The alternative is to perturb coordinates as follows:
# 
# $$
# \begin{aligned}
# h' =
# \begin{cases}
#     0 & \text{ with probability } p \\
#     \frac{h}{1-p} & \text{ otherwise}
# \end{cases}
# \end{aligned}
# $$
# 
# By design, the expectation remains unchanged, i.e. $\mathbf{E}[h'] = h$. This idea is at the heart of dropout where intermediate activations $h$ are replaced by a random variable $h'$ with matching expectation. The name 'dropout' arises from the notion that some neurons 'drop out' of the computation for the purpose of computing the final result. During training we replace intermediate activations with random variables

# In[ ]:


def dropout(X, drop_prob=0.3):
    assert 0 <= drop_prob <= 1
    # In this case, all elements are dropped out
    if drop_prob == 1:
        return torch.zeros_like(X)
    mask = torch.rand(*X.size()) > drop_prob
    # keep intermediate results unbiased
    return mask.type_as(X) * X / (1.0-drop_prob)


# ### Model with a Droput Layer

# In[ ]:


def net(x, method='relu'):
    x = x.view(x.shape[0], -1)
    hidden = activation(x.mm(w1), method=method)
    hidden = dropout(hidden)
    return hidden.mm(w2)


# ### Loss Function

# In[ ]:


loss_func = torch.nn.CrossEntropyLoss()


# ### Training

# In[ ]:


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 256, 784, 100, 10

# train_iter, test_iter = housing_data(batch_size)
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# Hyper-parameters
learning_rate = 1e-3
n_epochs = 10
display_freq = 50


for t in range(n_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = net(x, method='relu')
        

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the a scalar value held in the loss.
        
        loss = loss_func(y_pred, y)
        if batch_idx % display_freq == 0:
            print('epoch = {}\tbatch_idx = {}\tloss = {}'.format(t, batch_idx, loss.item()))

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()


# PyTorch: Standard APIs
# ----
# A fully-connected ReLU network with one hidden layer, trained to predict y from x
# by minimizing cross-entropy loss.
# 
# This implementation uses the nn package from PyTorch to build the network.
# PyTorch autograd makes it easy to define computational graphs and take gradients,
# but raw autograd can be a bit too low-level for defining complex neural networks;
# this is where the nn package can help. The nn package defines a set of Modules,
# which you can think of as a neural network layer that has produces output from
# input and may have some trainable weights.
# 
# **NOTICE**:
# In this section, we use built-in optimizer **SGD** with another hyper-parameter, i.e. momentum.

# ### Model using `nn` package

# In[ ]:


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 256, 784, 100, 10

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(H, D_out),
)


# ### Default Initialization
# 
# In the previous sections, e.g. in [“Concise Implementation of Linear Regression”](linear-regression-gluon.md), we used `net.initialize(init.Normal(sigma=0.01))` as a way to pick normally distributed random numbers as initial values for the weights. If the initialization method is not specified, such as `net.initialize()`, MXNet will use the default random initialization method: each element of the weight parameter is randomly sampled with an uniform distribution $U[-0.07, 0.07]$ and the bias parameters are all set to $0$. Both choices tend to work quite well in practice for moderate problem sizes.
# 
# ### Xavier Initialization
# 
# Let's look at the scale distribution of the activations of the hidden units $h_{i}$ for some layer. They are given by
# 
# $$h_{i} = \sum_{j=1}^{n_\mathrm{in}} W_{ij} x_j$$
# 
# The weights $W_{ij}$ are all drawn independently from the same distribution. Let's furthermore assume that this distribution has zero mean and variance $\sigma^2$ (this doesn't mean that the distribution has to be Gaussian, just that mean and variance need to exist). We don't really have much control over the inputs into the layer $x_j$ but let's proceed with the somewhat unrealistic assumption that they also have zero mean and variance $\gamma^2$ and that they're independent of $\mathbf{W}$. In this case we can compute mean and variance of $h_i$ as follows:
# 
# $$
# \begin{aligned}
#     \mathbf{E}[h_i] & = \sum_{j=1}^{n_\mathrm{in}} \mathbf{E}[W_{ij} x_j] = 0 \\
#     \mathbf{E}[h_i^2] & = \sum_{j=1}^{n_\mathrm{in}} \mathbf{E}[W^2_{ij} x^2_j] \\
#         & = \sum_{j=1}^{n_\mathrm{in}} \mathbf{E}[W^2_{ij}] \mathbf{E}[x^2_j] \\
#         & = n_\mathrm{in} \sigma^2 \gamma^2
# \end{aligned}
# $$
# 
# One way to keep the variance fixed is to set $n_\mathrm{in} \sigma^2 = 1$. Now consider backpropagation. There we face a similar problem, albeit with gradients being propagated from the top layers. That is, instead of $\mathbf{W} \mathbf{w}$ we need to deal with $\mathbf{W}^\top \mathbf{g}$, where $\mathbf{g}$ is the incoming gradient from the layer above. Using the same reasoning as for forward propagation we see that the gradients' variance can blow up unless $n_\mathrm{out} \sigma^2 = 1$. This leaves us in a dilemma: we cannot possibly satisfy both conditions simultaneously. Instead, we simply try to satisfy
# 
# $$
# \begin{aligned}
# \frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
# \sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}
# \end{aligned}
# $$
# 
# This is the reasoning underlying the eponymous Xavier initialization, proposed by [Xavier Glorot and Yoshua Bengio](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) in 2010. It works well enough in practice. For Gaussian random variables the Xavier initialization picks a normal distribution with zero mean and variance $\sigma^2 = 2/(n_\mathrm{in} + n_\mathrm{out})$.
# For uniformly distributed random variables $U[-a, a]$ note that their variance is given by $a^2/3$. Plugging $a^2/3$ into the condition on $\sigma^2$ yields that we should initialize uniformly with
# $U\left[-\sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}, \sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}\right]$.

# In[ ]:


torch.nn.init.xavier_normal_(model[0].weight)
torch.nn.init.xavier_normal_(model[-1].weight)


# In[ ]:


# The nn package also contains definitions of popular loss functions
loss_fn = torch.nn.CrossEntropyLoss()

# Hyper-parameters
learning_rate = 1e-3
momentum = 0.9
n_epochs = 10
display_freq = 50

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

for t in range(n_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        
        optimizer.zero_grad()
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x.view(x.shape[0], -1))

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y)
        if batch_idx % display_freq == 0:
            print('epoch = {}\tbatch_idx = {}\tloss = {}'.format(t, batch_idx, loss.item()))

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()
        
        optimizer.step()

