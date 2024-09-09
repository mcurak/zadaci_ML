import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from csv import reader
from numpy import array, float32, int32, newaxis
from cv2 import ml
from sklearn import model_selection as ms
from sklearn.metrics import accuracy_score, classification_report

# Load your data
data = pd.read_csv('C:/Users/matea/OneDrive/Desktop/listLabs-zadatak/training_data.csv')

# Assuming your CSV has columns: 'band1', 'band2', ..., 'ndvi', 'ndmi', 'label'
X = data[['Red', 'Green', 'Blue','NIR','ndvi','ndwi','msavi2','mtvi2', 'vari','tgi']]
y = data['Class']

# Split the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the data into training and temporary sets (70% training, 30% temporary)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Split the temporary set into validation and test sets (15% validation, 15% test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Get feature importances
importances = rf.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()

classification_rep = classification_report(y_test, y_pred)
print(classification_rep)