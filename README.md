# Gender Classification Throungh Image Using Convolutional Neural Networks

This project is a Convolutional Neural Network (CNN) model designed to classify images as male or female. The model has been trained on a custom dataset and achieves an accuracy of **96.97%** on the test set.

## Project Overview

The goal of this project is to develop a robust deep learning model that can accurately distinguish between male and female faces in images. The model is built using TensorFlow and Keras, leveraging the power of convolutional layers for feature extraction.

## Dataset

The dataset used for this project consists of labeled images of male and female faces. The images were preprocessed using the following steps:
- **Rescaling**: Pixel values were normalized by rescaling them to the range `[0, 1]`.
- **Augmentation**: Techniques like shearing, zooming, and horizontal flipping were applied to increase the diversity of the training data.

## Model Architecture

The CNN model is structured as follows:

1. **Convolutional Layers**: 
   - Filters: 32, 64
   - Kernel Size: 3x3
   - Activation: ReLU
   - Max Pooling: 2x2

2. **Flatten Layer**: Converts the 2D matrix into a vector.

3. **Fully Connected Layers**: 
   - Dense Layer: 128 units, Activation: ReLU
   - Dropout: 0.5 to prevent overfitting

4. **Output Layer**: 
   - Dense Layer: 1 unit, Activation: Sigmoid for binary classification

## Training

The model was trained using the following settings:
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 25

## Performance

The model achieves a test accuracy of **96.97%**, indicating a high level of performance for the gender classification task.

![gender_classification1](https://github.com/user-attachments/assets/03f56cfd-0999-431a-b729-00df20e6e12d)
![image](https://github.com/user-attachments/assets/2b9d4560-9852-4cfd-8801-bd256308426a)

## Installation

To run this project, you'll need to have Python 3.x and the following libraries installed:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

You can install the dependencies using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn
