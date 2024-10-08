Cassava Disease Classification using Enhanced DenseNet
This project implements a deep learning-based approach for classifying cassava leaf diseases using an enhanced version of the DenseNet architecture. Cassava is a major staple crop, and its health is vital to global food security. Accurately identifying cassava leaf diseases can help farmers take timely actions to protect their crops. In this project, DenseNet, a convolutional neural network (CNN) model, is fine-tuned and enhanced to improve performance on cassava disease classification.

Table of Contents
Introduction
Dataset
Model Architecture
Preprocessing
Training
Evaluation Metrics
Installation
Usage
Results
Contributors
License
Introduction
Cassava is a key crop in many parts of the world, particularly in Africa. However, cassava leaves are susceptible to various diseases that can significantly reduce crop yields. The goal of this project is to use an enhanced version of DenseNet to classify cassava leaf images into different disease categories. The model is trained on a dataset of cassava leaf images and can accurately predict the presence of diseases such as:

Cassava Bacterial Blight (CBB)
Cassava Brown Streak Disease (CBSD)
Cassava Green Mottle (CGM)
Cassava Mosaic Disease (CMD)
Healthy Leaf
Dataset
The dataset used for this project is the Cassava Leaf Disease Dataset, available on Kaggle. It contains images of cassava leaves with various disease labels.

Dataset Summary:
Cassava Bacterial Blight (CBB): Images of leaves affected by bacterial blight.
Cassava Brown Streak Disease (CBSD): Images showing brown streaks on leaves.
Cassava Green Mottle (CGM): Leaves with green mottle patterns caused by viral infection.
Cassava Mosaic Disease (CMD): Mosaic-like patterns on cassava leaves.
Healthy Leaf: Images of healthy cassava leaves.
Dataset Structure:
Image	Label
cassava_image_001.jpg	Cassava Bacterial Blight
cassava_image_002.jpg	Cassava Brown Streak Disease
cassava_image_003.jpg	Cassava Green Mottle
cassava_image_004.jpg	Cassava Mosaic Disease
cassava_image_005.jpg	Healthy Leaf
Source of Dataset: The dataset can be downloaded from Kaggle - Cassava Leaf Disease Dataset.

Model Architecture
The model is built using an enhanced version of the DenseNet architecture. DenseNet is a state-of-the-art CNN that efficiently connects each layer to every other layer to improve feature propagation and reduce vanishing gradients.

Enhancements:
Deeper Layers: Additional DenseNet blocks have been added to capture finer details of leaf patterns.
Regularization: Dropout and L2 regularization have been applied to prevent overfitting.
Custom Output Layer: The output layer is adjusted for the 5 categories in the cassava dataset.
Data Augmentation: Techniques like rotation, flipping, and zooming are used to improve model generalization.
DenseNet Model Code Example:
python
Copy code
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained DenseNet model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Regularization
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)  # 5 categories

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Preprocessing
Before training the model, the cassava leaf images need to be preprocessed:

Resizing: All images are resized to 224x224 pixels to match the input requirements of DenseNet.
Normalization: Pixel values are normalized to the range [0, 1].
Data Augmentation: Techniques like random rotations, flips, and zooms are used to improve the model’s generalization.
Example preprocessing code:

python
Copy code
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% for validation
)
Training
The enhanced DenseNet model is trained on the preprocessed dataset using early stopping to prevent overfitting.

Training Parameters:
Batch Size: 32
Epochs: 50
Optimizer: Adam optimizer with a learning rate of 0.001
Loss Function: Categorical Cross-Entropy
Metrics: Accuracy
Training example:

python
Copy code
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[early_stopping]
)
Evaluation Metrics
The performance of the model is evaluated using the following metrics:

Accuracy: The overall accuracy of the model in correctly classifying cassava leaf diseases.
Precision: The proportion of true positive predictions among all positive predictions.
Recall: The proportion of true positive predictions among all actual positive cases.
F1-Score: The harmonic mean of precision and recall.
Evaluation example:

python
Copy code
from sklearn.metrics import classification_report

y_pred = model.predict(test_generator)
y_pred_classes = y_pred.argmax(axis=-1)
print(classification_report(y_true, y_pred_classes, target_names=classes))
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/cassava-disease-classification-densenet.git
Navigate to the project directory:

bash
Copy code
cd cassava-disease-classification-densenet
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Dependencies
Python 3.x
TensorFlow 2.x
Keras
Numpy
Pandas
Scikit-learn
Matplotlib
OpenCV (optional, for image processing)
Usage
Prepare the dataset: Download the dataset from Kaggle or any other source and organize it into subfolders for each class.

Preprocess the images: Run the preprocessing script to resize and augment the images.

bash
Copy code
python preprocess.py --dataset_path /path/to/dataset
Train the model: Train the DenseNet model on the preprocessed dataset.

bash
Copy code
python train.py --epochs 50 --batch_size 32
Evaluate the model: Evaluate the model using the test data.

bash
Copy code
python evaluate.py --model_path /path/to/saved_model
Predict disease type: Use the trained model to classify new cassava leaf images.

bash
Copy code
python predict.py --image_path /path/to/image.jpg
