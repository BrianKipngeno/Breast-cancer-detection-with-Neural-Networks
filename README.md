# Breast-cancer-detection-with-Neural-Networks

This project uses the Breast Cancer Wisconsin dataset to build a neural network for classifying cancer diagnoses as either malignant (M) or benign (B). The dataset includes various features extracted from breast cancer cell images, such as radius, texture, perimeter, and more, which help in making the classification.

**Dataset**

The dataset is available here.

**Project Overview**

The project follows a series of steps to develop and evaluate a neural network model using PyTorch. The key steps include:

- **Data Preparation:** Loading the dataset, preprocessing it by removing unnecessary columns, encoding the diagnosis labels as numerical values (Malignant = 0, Benign = 1), and splitting the data into training and test sets.
- ** Building the Model:** Constructing a neural network with an input layer for the dataset's features, a hidden layer with - ReLU activation, and an output layer that predicts the diagnosis.
- **Training the Model:** The model is trained using cross-entropy loss and optimized using stochastic gradient descent (SGD). During training, forward propagation, loss computation, and backpropagation are performed iteratively over multiple epochs to minimize the loss function.
- **Evaluating the Model:**After training, the model is evaluated on the test set, and its accuracy is calculated to assess performance. Additionally, the loss trend is visualized across epochs to ensure proper model training.
  
**Installation**
To replicate this project, the following Python libraries are required:

- NumPy
- Pandas
- PyTorch
- Scikit-learn
- Seaborn
- Matplotlib
 
These can be installed using pip or another package manager.

**Steps Involved**

**1. Data Preparation**

The data is preprocessed to remove unnecessary columns, and the diagnosis column is encoded into binary values (0 for malignant, 1 for benign). The dataset is then split into training and testing sets to ensure the model can be evaluated on unseen data. Feature scaling is applied to standardize the data for optimal performance during training.

**2. Model Architecture**

The model is a simple feed-forward neural network with the following architecture:

- Input layer: Takes in the features from the dataset (30 in total).
- Hidden layer: A fully connected layer with 60 neurons and ReLU activation for non-linear transformations.
- Output layer: Two neurons representing the binary classification of the cancer diagnosis.
  
**3. Training the Model**

The model is trained over 100 epochs using the cross-entropy loss function and stochastic gradient descent (SGD) as the optimizer. Throughout the training process, loss values are recorded to track the model's progress. After each epoch, the optimizer updates the model weights based on the gradients computed during backpropagation.

**4. Model Evaluation**

Once the training is complete, the model's performance is assessed using the test set. Accuracy is calculated by comparing the model’s predictions with the actual test labels. The model achieves an accuracy of 98.25% on the test data, indicating strong performance in diagnosing breast cancer.

**5. Visualizing the Training Process**

The loss values from each epoch are plotted to visualize the model's learning progress. A decreasing loss trend indicates that the model is learning and improving with each iteration.

**Conclusion**

This project demonstrates how to use PyTorch to build and train a neural network for breast cancer detection. The model achieves a high accuracy of **98.25%**, making it a reliable tool for distinguishing between malignant and benign cancer cases. 

Future improvements may include experimenting with different network architectures or training techniques to further enhance performance.

**License**

This project is licensed under the MIT License.

