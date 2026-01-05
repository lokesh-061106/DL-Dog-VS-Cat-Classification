# 🐶🐱 Dog vs Cat Image Classification (Deep Learning)

This repository contains a **Deep Learning image classification project** that trains a model to distinguish between **dog and cat images**.  
The project uses a Convolutional Neural Network (CNN) based on MobileNetV2 architecture and demonstrates **transfer learning** using TensorFlow and Keras.

---

## 📌 Project Overview

Image classification is a core task in computer vision where a model learns to assign labels to images.  
In this project, we train a deep learning model that takes input images of size **224×224×3** and predicts whether the image contains a **dog** or a **cat**.

---

## 🧠 Technology and Tools Used

- **Python** – Main programming language  
- **TensorFlow & Keras** – Deep learning frameworks  
- **MobileNetV2** – Pre-trained CNN architecture for feature extraction  
- **Google Colab** – Development and training environment  
- **Jupyter Notebook** – Interactive code and documentation

---

## 🛠 Project Workflow

1. Import required libraries  
2. Load and preprocess dataset  
3. Build model using MobileNetV2 (Transfer Learning)  
4. Compile the model  
5. Train the model  
6. Evaluate model performance  
7. Make predictions on new images

---

## 📁 Dataset Structure
The dataset should be arranged in folders like:
train/
dogs/
cats/

test/
dogs/
cats/


Images are resized to **224×224** before training.

> You can use any Dog vs Cat dataset (for example from Kaggle or Google Drive).

---

## 🚀 Model Architecture

We used **MobileNetV2** as a feature extractor:

- Load pre-trained MobileNetV2 (ImageNet weights)
- Freeze its layers
- Add:
  - Global Average Pooling
  - Dense layer with softmax for 2 classes

This setup allows the model to benefit from features learned on large datasets while training faster.

---

## ▶️ How to Run the Project

1. Clone the repository:
   git clone https://github.com/lokesh-061106/DL-Dog-VS-Cat-Classification

2.Change directory:
cd DL-Dog-VS-Cat-Classification


3.Upload dataset to Colab or mount Google Drive:

4.Open the Notebook
DL-Dog-VS-Cat-Classification.ipynb

5.Run code cells step-by-step from top to bottom

📊 Model Training & Evaluation

The model uses Adam optimizer

Loss function: categorical crossentropy

Metrics tracked: accuracy

Plots of training and validation curves help visualize performance

Confusion matrix is used to check classification effectiveness

📌 Results

The trained model is capable of distinguishing between dogs and cats with good accuracy.
Training on a well-structured dataset improves performance and reduces overfitting.

⚡ Future Improvements

You can improve this project by:

Using data augmentation

Training with more data

Fine-tuning the MobileNet layers

Deploying the model as a web app

Adding real-time camera predictions

🤝 Acknowledgements

This project was developed for learning and academic purposes using open-source tools and datasets.

📝 Conclusion

This Dog vs Cat classification project demonstrates how deep learning and transfer learning can be applied to solve real computer vision tasks.
It provides hands-on experience in building CNN models using TensorFlow and Keras.

The dataset should be arranged in folders like:

