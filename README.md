<h2>Title:</h2>
<h1> Agricultural Market Image Classification Using Convolutional Neural Networks </h1>
<h2>Project Overview:</h2>
<ol>

This project implements a Deep Learning-based image classification system to automatically categorize agricultural market images into four major classes:

<strong>Indian Market</strong>
<li>

Onion

Potato

Tomato </li>

The model is built using Convolutional Neural Networks (CNNs) and is designed to assist in recognizing agricultural produce and market scenes from images.</ol>

<h2>Objective</h2>

<p>
To develop a CNN model capable of accurately classifying agricultural market images and analyzing the challenges associated with real-world imbalanced datasets.</p>

<strong> Dataset </strong>

<p>
The dataset consists of labeled agricultural images organized into training and testing folders.</p>
<pre>

Classes:

Indian Market

Onion

Potato

Tomato
</pre>

<h2>Folder Structure:</h2>
<pre>
dataset/
│
├── train/
│   ├── indian_market/
│   ├── onion/
│   ├── potato/
│   └── tomato/
│
└── test/
    ├── indian_market/
    ├── onion/
    ├── potato/
    └── tomato/
</pre>


The dataset is included in this repository as zip files and can be extracted for direct use.

<h2>Technologies Used:</h2>
<ol>
<li>
Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

Scikit-learn</li>


<h2>Model Architecture:</h2>

A custom CNN architecture with:
<ol>
<li>

3 Convolution + MaxPooling layers

Fully connected dense layers

Dropout regularization

Softmax output layer for multi-class classification</li>


<h2> Model Performance</h2>

<pre>
Training Accuracy: ~81%

Test Accuracy: ~74% 


The model performs well on dominant classes such as potato and market images, while the tomato class is limited due to very few training samples.</pre>

<h2>Evaluation Metrics:</h2>
<ul>
<li>
Accuracy

Precision

Recall

F1-score

Confusion Matrix </li> </ul>

These metrics provide detailed class-wise performance analysis.

<h2>Key Observations:</h2>

The dataset is highly imbalanced, which affects minority class (tomato) predictions.

The model generalizes well for major classes.

Misclassifications mainly occur between visually similar vegetables.

<h2> Future Improvements </h2>

<p>Collect more tomato images to balance the dataset

Apply transfer learning using pre-trained models (ResNet, MobileNet)

Use advanced augmentation techniques

Improve minority class recall</p>

<strong>How to Run</strong>
<ul>
<li>
Clone the repository

Extract the dataset zip files

Open the notebook

Run all cells sequentially</li><ul>