from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pickle

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into two classes
X_class1 = X[y < 2]
y_class1 = y[y < 2]
X_class2 = X[y >= 2]
y_class2 = y[y >= 2]

# Split the data for each class into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_class1, y_class1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_class2, y_class2, test_size=0.2, random_state=42)

# Train a KNN model for class 1 on client 1
knn_model1 = KNeighborsClassifier()
knn_model1.fit(X_train1, y_train1)

# Save the model for client 1
with open('model_client1.sav', 'wb') as file:
    pickle.dump(knn_model1, file)

# Train a KNN model for class 2 on client 2
knn_model2 = KNeighborsClassifier()
knn_model2.fit(X_train2, y_train2)

# Save the model for client 2
with open('model_client2.sav', 'wb') as file:
    pickle.dump(knn_model2, file)

# Evaluate and print metrics for client 1 model
y_pred1 = knn_model1.predict(X_test1)
print("Confusion Matrix for Client 1 Model:")
print(confusion_matrix(y_test1, y_pred1))
print("Accuracy for Client 1 Model:", accuracy_score(y_test1, y_pred1))

# Evaluate and print metrics for client 2 model
y_pred2 = knn_model2.predict(X_test2)
print("Confusion Matrix for Client 2 Model:")
print(confusion_matrix(y_test2, y_pred2))
print("Accuracy for Client 2 Model:", accuracy_score(y_test2, y_pred2))

# Load the models from the saved .sav files
with open('model_client1.sav', 'rb') as file:
    knn_model1_loaded = pickle.load(file)

with open('model_client2.sav', 'rb') as file:
    knn_model2_loaded = pickle.load(file)

# Aggregate the models to create a global model
def aggregate_models(models):
    # Initialize an empty model
    global_model = KNeighborsClassifier()
    
    # Concatenate the training data and labels of all models
    X_train_combined = np.concatenate([model._fit_X for model in models], axis=0)
    y_train_combined = np.concatenate([model._y for model in models], axis=0)
    
    # Fit the global model using the concatenated data
    global_model.fit(X_train_combined, y_train_combined)
    
    return global_model

# Aggregate the models
global_model = aggregate_models([knn_model1_loaded, knn_model2_loaded])

# Save the global model
with open('global_model.sav', 'wb') as file:
    pickle.dump(global_model, file)

# Evaluate and print metrics for the global model
X_test_combined = np.concatenate([X_test1, X_test2], axis=0)
y_true_global = np.concatenate([y_test1, y_test2], axis=0)
y_pred_global = global_model.predict(X_test_combined)

print("Confusion Matrix for Global Model:")
print(confusion_matrix(y_true_global, y_pred_global))
print("Accuracy for Global Model:", accuracy_score(y_true_global, y_pred_global))
