from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

# Display a sample digit
plt.figure(figsize=(4, 4))
plt.imshow(digits.images[0], cmap='gray')
plt.title(f"Label: {digits.target[0]}")
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Test with a custom example (optional)
index = np.random.randint(0, len(X_test))
sample_image = X_test[index].reshape(8, 8)
prediction = model.predict([X_test[index]])

plt.imshow(sample_image, cmap='gray')
plt.title(f"Predicted: {prediction[0]}, True: {y_test[index]}")
plt.show()
