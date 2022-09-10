<h2><center>Introduction to Deep Learning -Deep Neuron AI</center></h2>

Deep learning allows computational models that are composed of multiple processing **layers** to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains such as drug discovery and genomics. 

**Deep learning** is one of the leading tools in data analysis these days and one of the most common frameworks for deep learning is **Keras**. The Tutorial will provide an introduction to deep learning using `keras` with practical code examples.

## This Section will cover:

* Getting a conceptual understanding of multi-layer neural networks
* Training neural networks for image classification
* Implementing the powerful backpropagation algorithm
* Debugging neural network implementations

# Building Blocks: Artificial Neural Networks (ANN)

In machine learning and cognitive science, an artificial neural network (ANN) is a network inspired by biological neural networks which are used to estimate or approximate functions that can depend on a large number of inputs that are generally unknown. An ANN is built from nodes (neurons) stacked in layers between the feature vector and the target vector. A node in a neural network is built from Weights and Activation function.

An early version of ANN built from one node was called the **Perceptron**

<img src="https://github.com/ahammadmejbah/Introduction-to-Deep-Learning--Deep-Neuron-AI/blob/main/Images/perceptron.png" alt="perceptron">

The Perceptron is an algorithm for supervised learning of binary classifiers. functions that can decide whether an input (represented by a vector of numbers) belongs to one class or another. Much like logistic regression, the weights in a neural net are being multiplied by the input vector summed up and feeded into the activation function's input.

A Perceptron Network can be designed to have *multiple layers*, leading to the **Multi-Layer Perceptron** (aka `MLP`)

<img src="https://github.com/ahammadmejbah/Introduction-to-Deep-Learning--Deep-Neuron-AI/blob/main/Images/MLP.png" alt="MLP">



# Single Layer Neural Network
<img src="https://github.com/ahammadmejbah/Introduction-to-Deep-Learning--Deep-Neuron-AI/blob/main/Images/single%20layer%20nn.png" alt="SLNN">
_(Source: Python Machine Learning, S. Raschka)_


- We use a **gradient descent** optimization algorithm to learn the _Weights Coefficients_ of the model.
<br><br>
- In every **epoch** (pass over the training set), we update the weight vector $w$ using the following update rule:

$$
w = w + \Delta w, \text{where } \Delta w = - \eta \nabla J(w)
$$

<br><br>

In other words, we computed the gradient based on the whole training set and updated the weights of the model by taking a step into the **opposite direction** of the gradient $ \nabla J(w)$. 

In order to fin the **optimal weights of the model**, we optimized an objective function (e.g. the Sum of Squared Errors (SSE)) cost function $J(w)$. 

Furthermore, we multiply the gradient by a factor, the learning rate $\eta$ , which we choose carefully to balance the **speed of learning** against the risk of overshooting the global minimum of the cost function.


### Gradient Descent
In **gradient descent optimization**, we update all the **weights simultaneously** after each epoch, and we define the _partial derivative_ for each weight $w_j$ in the weight vector $w$ as follows:

$$
\frac{\partial}{\partial w_j} J(w) = \sum_{i} ( y^{(i)} - a^{(i)} )  x^{(i)}_j
$$

**Note**: _The superscript $(i)$ refers to the i-th sample. The subscript $j$ refers to the j-th dimension/feature_


Here $y^{(i)}$ is the target class label of a particular sample $x^{(i)}$ , and $a^{(i)}$ is the **activation** of the neuron 

(which is a linear function in the special case of _Perceptron_).

We define the **activation function** $\phi(\cdot)$ as follows:

$$
\phi(z) = z = a = \sum_{j} w_j x_j = \mathbf{w}^T \mathbf{x}
$$

## Binary Classification
While we used the **activation** $\phi(z)$ to compute the gradient update, we may use a **threshold function** _(Heaviside function)_ to squash the continuous-valued output into binary class labels for prediction:

$$
\hat{y} = 
\begin{cases}
    1 & \text{if } \phi(z) \geq 0 \\
    0 & \text{otherwise}
\end{cases}
$$

## Building Neural Nets from scratch 

### Idea:

We will build the neural networks from first principles. 
We will create a very simple model and understand how it works. We will also be implementing backpropagation algorithm. 

**Please note that this code is not optimized and not to be used in production**. 

This is for instructive purpose - for us to understand how ANN works. 

Libraries like `theano` have highly optimized code.

### Perceptron and Adaline Models

Take a look at this notebook : <a href="1.1.1 Perceptron and Adaline.ipynb" target="_blank_"> Perceptron and Adaline </a>

If you want a sneak peek of alternate (production ready) implementation of _Perceptron_ for instance try:
```python
from sklearn.linear_model import Perceptron
```

## Introducing the multi-layer neural network architecture
<img src="https://github.com/ahammadmejbah/Introduction-to-Deep-Learning--Deep-Neuron-AI/blob/main/Images/mln.png" width="50%" />

_(Source: Python Machine Learning, S. Raschka)_


Now we will see how to connect **multiple single neurons** to a **multi-layer feedforward neural network**; this special type of network is also called a **multi-layer perceptron** (MLP). 

The figure shows the concept of an **MLP** consisting of three layers: one _input_ layer, one _hidden_ layer, and one _output_ layer. 

The units in the hidden layer are fully connected to the input layer, and the output layer is fully connected to the hidden layer, respectively. 

If such a network has **more than one hidden layer**, we also call it a **deep artificial neural network**.



we denote the `ith` activation unit in the `lth` layer as $a_i^{(l)}$ , and the activation units $a_0^{(1)}$ and 
$a_0^{(2)}$ are the **bias units**, respectively, which we set equal to $1$. 
<br><br>
The _activation_ of the units in the **input layer** is just its input plus the bias unit:

$$
\mathbf{a}^{(1)} = [a_0^{(1)}, a_1^{(1)}, \ldots, a_m^{(1)}]^T = [1, x_1^{(i)}, \ldots, x_m^{(i)}]^T
$$
<br><br>
**Note**: $x_j^{(i)}$ refers to the jth feature/dimension of the ith sample


### Notes on Notation (usually) Adopted

The terminology around the indices (subscripts and superscripts) may look a little bit confusing at first. 
<br><br>

You may wonder why we wrote $w_{j,k}^{(l)}$ and not $w_{k,j}^{(l)}$ to refer to 
the **weight coefficient** that connects the *kth* unit in layer $l$ to the jth unit in layer $l+1$. 
<br><br>

What may seem a little bit quirky at first will make much more sense later when we **vectorize** the neural network representation. 
<br><br>

For example, we will summarize the weights that connect the input and hidden layer by a matrix 
$$ W^{(1)} \in \mathbb{R}^{h×[m+1]}$$

where $h$ is the number of hidden units and $m + 1$ is the number of hidden units plus bias unit. 

<img src="https://github.com/ahammadmejbah/Introduction-to-Deep-Learning--Deep-Neuron-AI/blob/main/Images/ml2.png" width="50%" />

_(Source: Python Machine Learning, S. Raschka)_

## Forward Propagation

* Starting at the input layer, we forward propagate the patterns of the training data through the network to generate an output.

* Based on the network's output, we calculate the error that we want to minimize using a cost function that we will describe later.


### Sigmoid Activation
<img src="https://github.com/ahammadmejbah/Introduction-to-Deep-Learning--Deep-Neuron-AI/blob/main/Images/sig.png" width="50%" />

_(Source: Python Machine Learning, S. Raschka)_



* We backpropagate the error, find its derivative with respect to each weight in the network, and update the model.

### Our neural networks class

When we first create a neural networks architecture, we need to know the number of inputs, number of hidden layers and number of outputs.The weights have to be randomly initialized.

```python
class MLP:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        
        # set them to random vaules
        self.wi = rand(-0.2, 0.2, size=self.wi.shape)
        self.wo = rand(-2.0, 2.0, size=self.wo.shape)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
```

### Activation Function

```python
def activate(self, inputs):
        
    if len(inputs) != self.ni-1:
        print(inputs)
        raise ValueError('wrong number of inputs')

    # input activations
    for i in range(self.ni-1):
        self.ai[i] = inputs[i]

    # hidden activations
    for j in range(self.nh):
        sum_h = 0.0
        for i in range(self.ni):
            sum_h += self.ai[i] * self.wi[i][j]
        self.ah[j] = sigmoid(sum_h)

    # output activations
    for k in range(self.no):
        sum_o = 0.0
        for j in range(self.nh):
            sum_o += self.ah[j] * self.wo[j][k]
        self.ao[k] = sigmoid(sum_o)

    return self.ao[:]
```


### BackPropagation


```python
def backPropagate(self, targets, N, M):
        
    if len(targets) != self.no:
        print(targets)
        raise ValueError('wrong number of target values')

    # calculate error terms for output
    output_deltas = np.zeros(self.no)
    for k in range(self.no):
        error = targets[k]-self.ao[k]
        output_deltas[k] = dsigmoid(self.ao[k]) * error

    # calculate error terms for hidden
    hidden_deltas = np.zeros(self.nh)
    for j in range(self.nh):
        error = 0.0
        for k in range(self.no):
            error += output_deltas[k]*self.wo[j][k]
        hidden_deltas[j] = dsigmoid(self.ah[j]) * error

    # update output weights
    for j in range(self.nh):
        for k in range(self.no):
            change = output_deltas[k] * self.ah[j]
            self.wo[j][k] += N*change + 
                             M*self.co[j][k]
            self.co[j][k] = change

    # update input weights
    for i in range(self.ni):
        for j in range(self.nh):
            change = hidden_deltas[j]*self.ai[i]
            self.wi[i][j] += N*change + 
                             M*self.ci[i][j]
            self.ci[i][j] = change

    # calculate error
    error = 0.0
    for k in range(len(targets)):
        error += 0.5*(targets[k]-self.ao[k])**2
    return error
```
