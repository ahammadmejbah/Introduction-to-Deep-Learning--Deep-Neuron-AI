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

The Perceptron is an algorithm for supervised learning of binary classifiers. functions that can decide whether an input (represented by a vector of numbers) belongs to one class or another. Much like logistic regression, the weights in a neural net are being multiplied by the input vector summed up and feeded into the activation function's input.

A Perceptron Network can be designed to have *multiple layers*, leading to the **Multi-Layer Perceptron** (aka `MLP`)



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







