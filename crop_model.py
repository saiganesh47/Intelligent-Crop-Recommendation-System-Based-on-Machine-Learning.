from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection

# Load the dataset
crop = pd.read_csv('Data/crop_recommendation.csv')

# Separate features and labels
X = crop.iloc[:, :-1].values
Y = crop.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Define the models for the voting classifier
models = []
models.append(('SVC', SVC(gamma='auto', probability=True)))
models.append(('svm1', SVC(probability=True, kernel='poly', degree=1)))
models.append(('svm2', SVC(probability=True, kernel='poly', degree=2)))
models.append(('svm3', SVC(probability=True, kernel='poly', degree=3)))
models.append(('svm4', SVC(probability=True, kernel='poly', degree=4)))
models.append(('svm5', SVC(probability=True, kernel='poly', degree=5)))
models.append(('rf', RandomForestClassifier(n_estimators=100, random_state=42)))  # Improved RandomForest
models.append(('gnb', GaussianNB()))
models.append(('knn1', KNeighborsClassifier(n_neighbors=1)))
models.append(('knn3', KNeighborsClassifier(n_neighbors=3)))
models.append(('knn5', KNeighborsClassifier(n_neighbors=5)))
models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
models.append(('knn9', KNeighborsClassifier(n_neighbors=9)))

# Initialize Voting Classifier (Soft Voting)
vot_soft = VotingClassifier(estimators=models, voting='soft')

# Train the Voting Classifier
vot_soft.fit(X_train, y_train)

# Make predictions
y_pred = vot_soft.predict(X_test)

# Evaluate the model
scores = model_selection.cross_val_score(vot_soft, X_train, y_train, cv=5, scoring='accuracy')  # Use training data for cross-validation
print("Cross-Validation Accuracy: ", scores.mean())

# Calculate accuracy on the test set
score = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Test Accuracy: {score * 100:.2f}%")

# Save the model using pickle
import pickle
pkl_filename = 'Crop_Recommendation.pkl'
with open(pkl_filename, 'wb') as Model_pkl:
    pickle.dump(vot_soft, Model_pkl)

print("Model saved successfully.")
