# Machine Learning Course Repository

This repository contains code and resources related to the labs for the CPSC_5616-Machine Learning/Deep Learning course(2024W) at Laurentian University. The course covers a wide range of topics in machine learning, deep learning, neural network and more.

## Purpose

The purpose of this repository is to provide a centralized location for accessing and managing course materials, including lecture notes, assignments, and project specifications.

## Jupyter Notebook Setup

To run the code provided in the Jupyter Notebook files within this repository, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/Airmacho/CPSC_5616_MLDL.git
    ```

2. Navigate to the directory containing the Jupyter Notebook file for the specific assignment or project you want to run.

3. Launch Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

4. In your web browser, open the Jupyter Notebook file (.ipynb) and execute the code cells sequentially.

## Assignment 1: Linear Regression

The first assignment focuses on implementing linear regression, a fundamental technique in machine learning, using Python and Jupyter Notebook. In this assignment, you will:

- Load and preprocess a dataset
- Implement the gradient descent algorithm to optimize the parameters of a linear regression model with one variable
- Evaluate the performance of the model using appropriate test cases
- Visualize the results and analyze the relationship between input features and target variable

To access the assignment notebook, navigate to the `Assignment_1` directory and open the `C1_W2_Linear_Regression.ipynb` file.

## Assignment 2: Linear Regression

The second assignment focuses on using neural networks for handwritten digit recognition, specifically digits 0-9.

- Introduced the Rectified Linear Unit (ReLU) activation function for non-linearity in neural networks.
- Utilized the softmax function to convert output values into a probability distribution for multiclass classification.
- Implemented a neural network for recognizing handwritten digits 0-9 using TensorFlow.
- Worked with a dataset of 5000 training examples of 20x20 grayscale images unrolled into a 400-dimensional vector.
- Demonstrated creating a NumPy implementation of the softmax function for converting values to a probability distribution.
- Examined the neural network model representation with two dense layers and an output layer tailored for digit recognition.

To access the assignment notebook, navigate to the `Assignment_2` directory.

## Assignment 3: Decision Trees

The third assignment centers around creating a decision tree using the provided dataset. The procedure for constructing the decision tree is detailed as follows:

1. Entropy Calculation:
Develop a utility function named calculate_entropy to evaluate the entropy (representing impurity) at a given node.

2. Dataset Partitioning:
Establish a function named partition_dataset to segment the data at a node into left and right branches based on a chosen feature.

3. Information Gain Computation:
Implement a function to determine the information gain achieved by splitting on a particular feature.

4. Feature Selection:
Identify the feature that yields the maximum information gain to effectively partition nodes and construct the decision tree.

## Assignment 4: K-means Clustering

The assignment introduces the K-means algorithm for clustering data points together. It involves starting with a sample dataset to understand the algorithm and then using it for image compression by reducing the number of colors in an image.

- **Implementing K-means**:
  - The algorithm iteratively assigns data points to centroids and refines these assignments based on the mean of the points assigned to each centroid.
  - The process involves initializing centroids, assigning examples to the closest centroids, and updating centroids based on these assignments.

- **Finding Closest Centroids**:
  - In this phase, each training example is assigned to its closest centroid based on Euclidean distance.
  - The function `find_closest_centroids` computes these assignments by minimizing the Euclidean distance between data points and centroids.

- **Computing Centroid Means**:
  - After assigning points to centroids, the algorithm recomputes the mean of points assigned to each centroid.
  - The function `compute_centroids` calculates new centroids by computing the mean of points assigned to each centroid.

- **K-means on a Sample Dataset**:
  - The K-means algorithm is applied to a toy 2D dataset after implementing the two functions.
  - The algorithm runs iteratively, visualizing the progress at each iteration to demonstrate how K-means works effectively.