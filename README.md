# CAT & DOG PREDICTION

- Written by Group 193 - Vien Dat Huynh (z5223470)
- School of Computer Science and Engineering, University of New South Wales


## 1. PROJECT OVERVIEW
Throughout the exploration in the course COMP9417 - Machine Learning, the Cat & Dog Prediction project develops two models for object detection: Logistic Regression and Convolution Neural Network. Being trained and tested using the dataset Dogs vs. Cats from Kaggle, the two model’s performances and outputs are analysed and illustrated with graphs.

The project implements two different models from basic to advanced:

Logistic Regression: By converting the images into 1D dataset, they can be fitted with this model and generate discrete outputs based on probability calculations.

Convolutional Neural Network: Being a concept of Deep Learning where all the hypes about AI come from, CNN is often considered as one of the most effective models for classifying images.
After building these two models, they will be trained and tested with datasets of different sizes. Regarding the CNN model, several popular activation functions and optimizers will also be tested and we will compare their performances.

Overall, the Regression model shows a below average performance with accuracy of only about 50% while the CNN model generates predictions with acceptable accuracy of over 70%. This could be because they are fitted with only images of size 50x50 due to the lack of powerful hardware. It is clear that CNN models outperforms Regression in the problem of image classification.

The related files of this project could be found in the links below:
- Kaggle: https://www.kaggle.com/c/dogs-vs-cats (only the dataset)
- Github: https://github.com/datvienhuynh/dogs-cats-prediction (all related files include Python Notebook)


## 2. DATASET
The two models are trained and tested with the Dogs vs. Cats dataset from Kaggle Competition
https://www.kaggle.com/c/dogs-vs-cats/
which contains 12,500 dog images, 12,500 cat images and 12,500 unlabeled images. Due to the time limit of the project, only the labeled dataset is utilised to train and test the machine learning models. A couple of unknown samples will be manually tested for tester's observation.

## 3. IMPLEMENTATION
### Directory Dataset
Before processing images, two TensorFlow datasets (labeled and unlabeled) are created containing JPG file paths to two folders 'dogs-vs-cats/train' and 'dogs-vs-cats/test1'

### Image Setting
Modify variables below to determine the quantity and quality of images for training and testing:
- DATASET_SIZE: Control the number of images pulled out from the directory dataset. Modify its value in range [100, 25000] based on the training and testing purpose
- REGRESSION_DATASET: Setup dataset size for training Regression model. REGRESSION_DATASET <= DATASET_SIZE
- UNLABELED_CHECK_SAMPLES: Setup the number of unlabeled images for manually checking. UNLABELED_CHECK_SAMPLES <= 25,000
- EVALUATE_SAMPLES: Setup dataset size for evaluating activation functions/optimizers. EVALUATE_SAMPLES <= DATASET_SIZE
- IMG_HEIGHT, IMG_WIDTH: Setup the image resolution. 50x50 - 100x100 is recommended, everything above 200x200 will result in a significant amount of training time or even kernel's death.

### Process Train & Test Dataset
Convert the TensorFlow datasets to a vector form of grayscale Tensor images and labels.

- Use Dataset.map to create a dataset of Tensor (image, label) pairs. The label of each image is created based on its name such as 'dog.123.jpg' and 'cat.123.jpg'.
- Convert and flattem images into vector forms to fit the Regression model.
- Convert images to grayscale (1 dimension) and Numpy array to fit the CNN model.
- Convert labels to One Hot vectors 0: [1. 0.] (Dog) and 1: [0. 1.] (Cat)

### Build Logistic Regression Model
- Use Logistic Regression( ) model of sklearn
- Fit and test the model with given number of samples
- Manually test the model with unlabeled dataset

### Build Convolutional Neural Network
- Use the model Sequential() and setup hidden layers of the network
- 2 convolutional layers of kernel (3, 3) extracts a feature map from the image data
- Pooling layer reduces the number of featured values and therefore training time
- Dropping layer is also added to prevent overfitting
- Flattening layer converts the feature maps into a vector of values that can be passed to fully connected layers
- Fully connected layer generates the prediction for images
- The model is compiled with activation function relu and optimizer Adam

## 4. PERFORMANCE

### Performance of the Regression model with dataset of different sizes:
- Size: 100, Image Resolution: 50 x 50, Train Set Accuracy = 100%, Test Set Evaluation: accuracy = 50%
- Size: 1000, Image Resolution: 50 x 50, Train Set Accuracy = 100%, Test Set Evaluation: accuracy = 44%
- Size: 5000, Image Resolution: 50 x 50, Train Set Accuracy = 95.28%, Test Set Evaluation: accuracy = 53.2%
- Size: 10000, Image Resolution: 50 x 50, Train Set Accuracy = 78.24%, Test Set Evaluation: accuracy = 50.3%

### Performance of the CNN model with dataset of different sizes:
- Size: 10000, Image Resolution: 50 x 50, epochs = 15
Test Set Evaluation: loss = 1.247293, accuracy = 69.20

- Size: 15000, Image Resolution: 50 x 50, epochs = 15
Test Set Evaluation: loss = 1.051655, accuracy = 68.20

- Size: 20000, Image Resolution: 50 x 50, epochs = 15
Test Set Evaluation: loss = 0.946988, accuracy = 69.45

- Size: 25000, Image Resolution: 50 x 50, epochs = 15
Test Set Evaluation: loss = 0.883547, accuracy = 71.28
