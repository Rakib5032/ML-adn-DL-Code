import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Define output directory and ensure it exists
output_dir = 'D:\\NewModel\\Model\\Model_Results'
os.makedirs(output_dir, exist_ok=True)

# Function to plot and save loss and accuracy curves
def plot_loss_accuracy_curves(history, output_dir):
    curve_dir = os.path.join(output_dir, "curves")
    os.makedirs(curve_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(curve_dir, 'loss_accuracy_curves.png'))
    plt.close()

# Function to save confusion matrix and classification metrics
def save_classification_metrics(model, data, output_dir, dataset_name):
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    true_labels = data.classes
    predictions = model.predict(data, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.class_indices.keys(), 
                yticklabels=data.class_indices.keys())
    plt.title(f'Confusion Matrix ({dataset_name})')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.savefig(os.path.join(metrics_dir, f'confusion_matrix_{dataset_name}.png'))
    plt.close()

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=data.class_indices.keys(), 
                         columns=data.class_indices.keys())
    cm_df.to_csv(os.path.join(metrics_dir, f'confusion_matrix_{dataset_name}.csv'))

    # Classification report
    report = classification_report(true_labels, predicted_labels, target_names=data.class_indices.keys(), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(metrics_dir, f'classification_report_{dataset_name}.csv'))

    print(f"{dataset_name} Classification Metrics Saved!")

# Function to calculate and plot error rate
def plot_error_rate(history, output_dir):
    error_rate_dir = os.path.join(output_dir, "error_rate")
    os.makedirs(error_rate_dir, exist_ok=True)

    train_error = 1 - np.array(history.history['accuracy'])
    val_error = 1 - np.array(history.history['val_accuracy'])

    plt.figure(figsize=(8, 6))
    plt.plot(train_error, label='Training Error Rate', color='red')
    plt.plot(val_error, label='Validation Error Rate', color='blue')
    plt.title('Error Rate Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(error_rate_dir, 'error_rate_curve.png'))
    plt.close()

# Data Preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_datagen.flow_from_directory(
    r'D:\\NewModel\\Model\\train',  # Adjust path as needed
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_data = valid_datagen.flow_from_directory(
    r'D:\\NewModel\\Model\\val',  # Adjust path as needed
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    r'D:\\NewModel\\Model\\test',  # Adjust path as needed
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Model Building
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
predictions = layers.Dense(len(train_data.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model Training
history_object = model.fit(train_data, validation_data=valid_data, epochs=15, batch_size=32)

# Save the model
model.save(os.path.join(output_dir, 'densenet201_model.keras'))

# Save the training history
history_dir = os.path.join(output_dir, "history")
os.makedirs(history_dir, exist_ok=True)

with open(os.path.join(history_dir, 'history.pkl'), 'wb') as f:
    pickle.dump(history_object.history, f)

# Plot and save loss and accuracy curves
plot_loss_accuracy_curves(history_object, output_dir)

# Save confusion matrix and metrics for train data
save_classification_metrics(model, train_data, output_dir, "train")

# Save confusion matrix and metrics for validation data
save_classification_metrics(model, valid_data, output_dir, "val")

# Save confusion matrix and metrics for test data
save_classification_metrics(model, test_data, output_dir, "test")

# Plot and save error rate
plot_error_rate(history_object, output_dir)

print(f'All results saved in {output_dir}')
