import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load Iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 16
EPOCHS = 50
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
OPTIMIZER = 'sgd'

# Build ANN model
print("\nBuilding ANN model...")
model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(16, activation='relu', name='hidden_layer_1'),
    layers.Dense(8, activation='relu', name='hidden_layer_2'),
    layers.Dense(3, activation='softmax', name='output_layer')
])

# Compile model with specified hyperparameters
optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=LOSS_FUNCTION,
    metrics=['accuracy']
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
print("\nEvaluating model on test set...")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
print("\nSaving model...")
model.save('iris_ann_model.h5')

# Save training history
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']]
}

with open('training_history.json', 'w') as f:
    json.dump(history_dict, f)

# Save model metadata
metadata = {
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist(),
    'hyperparameters': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'loss_function': LOSS_FUNCTION,
        'optimizer': OPTIMIZER
    },
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss)
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("\nModel training complete!")
print("Files saved:")
print("  - iris_ann_model.h5 (trained model)")
print("  - scaler.pkl (feature scaler)")
print("  - training_history.json (training metrics)")
print("  - model_metadata.json (model information)")