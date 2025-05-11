<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>NAME: VINCY JOVITHA V</H3>
<H3>REGISTER NO.: 212223230242</H3>
<H3>EX. NO.5</H3>
<H3>DATE: 09.05.2025</H3>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>

<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>

<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

<H3>PROGRAM:</H3>

```py
import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, mu, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x - mu)**2)

def end_to_end(X1, X2, ys, mu1, mu2):
    # Transform data points using RBF
    transformed_1 = [gaussian_rbf(np.array([X1[i], X2[i]]), mu1) for i in range(len(X1))]
    transformed_2 = [gaussian_rbf(np.array([X1[i], X2[i]]), mu2) for i in range(len(X1))]

    plt.figure(figsize=(13, 5))

    # Plot original data (non-linearly separable)
    plt.subplot(1, 2, 1)
    plt.scatter(X1[:2], X2[:2], label="Class_0")
    plt.scatter(X1[2:], X2[2:], label="Class_1")
    plt.xlabel("$X1$", fontsize=15)
    plt.ylabel("$X2$", fontsize=15)
    plt.title("Xor: Linearly Inseparable", fontsize=15)
    plt.legend()

    # Plot transformed data (linearly separable)
    plt.subplot(1, 2, 2)
    plt.scatter(transformed_1[:2], transformed_2[:2], label="Class_0")
    plt.scatter(transformed_1[2:], transformed_2[2:], label="Class_1")
    plt.plot([0, 0.95], [0.95, 0], "k--")
    plt.annotate("Seperating hyperplane", xy=(0.4, 0.55), xytext=(0.55, 0.66),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel(f"$mu1$: {mu1}", fontsize=15)
    plt.ylabel(f"$mu2$: {mu2}", fontsize=15)
    plt.title("Transformed Inputs: Linearly Seperable", fontsize=15)
    plt.legend()

    # Linear transformation (solving for weights)
    A = np.column_stack([transformed_1, transformed_2, np.ones(len(X1))])
    W = np.linalg.inv(A.T @ A) @ A.T @ ys
    print(f"Predicted: {np.round(A @ W)}")
    print(f"True Labels: {ys}")
    print(f"Weights: {W}")
    return W

def predict_matrix(point, weights, mu1, mu2):
    # RBF transformation of input point
    transformed_0 = gaussian_rbf(point, mu1)
    transformed_1 = gaussian_rbf(point, mu2)
    A = np.array([transformed_0, transformed_1, 1])
    return np.round(A @ weights)

# Data points
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
ys = np.array([0, 1, 1, 0])

# Centers of RBF
mu1 = np.array([0, 1])
mu2 = np.array([1, 0])

# Train model
weights = end_to_end(x1, x2, ys, mu1, mu2)

# Testing predictions
for test_point in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(f"Input: {test_point}, Predicted: {predict_matrix(np.array(test_point), weights, mu1, mu2)}")

```

<H3>OUTPUT:</H3>

![image](https://github.com/user-attachments/assets/d8524aab-a191-4f0a-b67d-8b9be88e01a8)

<H3>Result:</H3>
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.
