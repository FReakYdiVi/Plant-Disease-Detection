# **Plant-Disease-Detection using transfer learning** 

This project aims to detect plant diseases using deep learning techniques, specifically transfer learning with VGG-16 and ResNet-50 architectures. Transfer learning allows us to leverage pre-trained models on large datasets and fine-tune them for specific tasks.

### **Transfer Learning**

Transfer learning is a machine learning technique where a model trained on one task is adapted for a related task. In this project, we use transfer learning to leverage the knowledge learned by models trained on large image datasets (such as ImageNet) and apply it to the task of plant disease detection.

## Models Used
### VGG-16
VGG-16 is a convolutional neural network architecture known for its simplicity and effectiveness. We fine-tuned the last layers of the VGG-16 model for our plant disease detection task.

### ResNet-50
ResNet-50 is a deeper convolutional neural network architecture that addresses the vanishing gradient problem by introducing skip connections. We also fine-tuned the last layers of the ResNet-50 model for our plant disease detection task.

## Model Explanation

 There is addition of two dense layers with transfer learning models with 64 neurons and activation function **relu** and second with three neurons and **softmax** activation function for last dense layer and fine tuning the last layers of both models.

# Results
- The VGG-16 model achieved a highest accuracy of 99.3% on the validation set layers.
- The ResNet-50 model achieved a highest accuracy of 85.2% on the validation set.

#### References 

Our model is referenced from this [reasearch paper](https://www.biorxiv.org/content/10.1101/2020.05.22.110957v2.full.pdf) mostly