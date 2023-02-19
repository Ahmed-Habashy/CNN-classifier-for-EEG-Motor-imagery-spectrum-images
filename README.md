This Python script creates, trains, and tests a Convolutional Neural Network (CNN) for image classification using various libraries like Numpy, Tensorflow, OpenCV, Keras, etc. The input images are spectrum images that are loaded from a specified folder path and pre-processed by resizing and normalizing.

# Prerequisites
The following libraries and dependencies are required to run the script:

Numpy
Tensorflow
OpenCV
Keras
matplotlib
sklearn
os
sys
matplotlib.image
PIL

# How to use
Ensure that the above-listed libraries are installed on your system.
Download and save the script to your local machine.
Open the script in a Python environment (e.g., Anaconda, Jupyter Notebook).
Update the img_folder variable to reflect the path of the input spectrum images.
Update the IMG_WIDTH and IMG_HEIGHT variables to reflect the required size of the input images.
Update the subject variable to reflect the name of the subject for which the model is being built.
Run the script.

# Output
The script generates a CNN model for image classification based on the input spectrum images. The output includes the raw data and test data sizes, as well as the accuracy of the model for each fold.



