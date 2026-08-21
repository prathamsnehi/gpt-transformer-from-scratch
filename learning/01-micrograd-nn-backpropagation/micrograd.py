# things required for micrograd to work:

import math
import numpy as np
import matplotlib.pyplot as plt
import random

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # initially, this node doesn't affect the final value (therefore 0 gradient)
        self._backward = lambda: None # by default, nothing to do (for leaf node)
        
        self._prev = set(_children) # who is contributing to this value?
        self._op = _op # what operation created this value?
        self.label = label # label to represent in like visualizing graphs

    def __repr__(self): # python uses __repr__ internally to return a string or something as a representation
        # instead of just showing location in memory
        return f"Value(data={self.data})"

    def __add__(self, other): # special method to define the plus operator for this object:
        # a + b will be a.__add__(b), and b will be passed as other

        if not isinstance(other, Value):
            other = Value(other) # convert to Value node if an int/float is flowing through
         
        out = Value(self.data + other.data, (self, other), '+') # making self and other as prev of this value

        def _backward():
            self.grad += 1.0 * out.grad # local derivative * global der of out
            other.grad += 1.0 * out.grad # same logic

        out._backward = _backward # assigning it this function
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other) # # convert to Value node if an int/float is flowing through
        
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad # local * global of out
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            # we already have
            self.grad += (1-t**2) * out.grad # local * global of out
        
        out._backward = _backward
        return out
        
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self) # builds it from here and back

        self.grad = 1.0 # assuming this is the last thing
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, num_inputs: int) -> None:
        self.weights = []
        self.bias = Value(random.uniform(-1, 1))

        # initializing weights randomly at first:
        for i in range(num_inputs):
            self.weights.append(Value(random.uniform(-1, 1)))

    def __call__(self, x):
        # imagine you created an object n from Neuron
        # when you do n(x), this function is called, and x can be any input

        # we need to return the forward pass of this neuron, i.e.
        # Summation(wixi) + b
        # Therefore, we expect x to be an array of the same size as num_inputs
        activation = sum(wi*xi for wi, xi in zip(self.weights, x)) + self.bias
        return activation.tanh()

class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neurons = []
        for i in range(num_outputs): # each output is a Neuron in this layer, so:
            self.neurons.append(Neuron(num_inputs))

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        if len(outs) == 1:
            return outs[0]
        return outs    

class MLP:
    def __init__(self, num_inputs, num_output_list):
        """
        num inputs: number of input neurons
        num_output_list: a list containing number of neurons in each layer after the input layer
        """
        sizes = [num_inputs] + num_output_list # sizes of all the layers in the MLP
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i+1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x