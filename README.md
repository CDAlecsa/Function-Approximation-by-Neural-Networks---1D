# Function-Approximation-by-Neural-Networks---1D

This repository contains 3 folders : the first one using neural networks from scratch (a simplified version using the code from the repository 'DNN for MNIST dataset'), the second one using Keras and the third one using Tensorflow.
In the first two folders, the files with the name 'Discretization' and 'Discretization_2' contains approximation functions by neural networks, where the mesh is divided in sub-intervals.
Also, in 'Discretization'-type files, one has weights that are initialized independently on each sub-interval,
while in 'Discretization_2'-type files, the weights are reused after the neural network training on the previous sub-interval.


Notes:
- I have put some new activation functions (gaussian, gaussian wavelet, Morlet)
- plots for the accuracy on each interval
- plots for the absolute errors between the function and the approximation
- the numbers on the dataset are generated uniformly
- if the error is high, then one needs to normalize the function such that it takes values between 0 and 1 and also rescale the interval to [0,1] or [-1,1]
- the only admissible loss function is the quadratic cost function
- the activation function between the last 2 layers is the identity / linear function

Future implementation:
- In Tensorflow, I will make some additional files in which I will use the technique of dividing the mesh into sub-intervals.
