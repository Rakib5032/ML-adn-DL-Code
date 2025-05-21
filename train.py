import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory paths (adjust paths as needed)
train_dir = r'D:\\NEW MODEL\\Model\\train'
val_dir = r'D:\\NEW MODEL\\Model\\val'
test_dir = r'D:\\NEW MODEL\\Model\\test'
output_dir = r'D:\\NEW MODEL\\Model\\output'

# Hyperparameters
batch_size = 32
img_size = (224, 224)
learning_rate = 0.001
num_epochs = 20

# Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load pre-trained DenseNet201 model and modify for custom classification
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model to prevent updates during initial training

# Custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(len(train_data.class_indices), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Training the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=num_epochs,
    callbacks=[early_stopping],
    verbose=1
)

# Create output directories
test_metrics_dir = os.path.join(output_dir, "test_metrics")
os.makedirs(test_metrics_dir, exist_ok=True)

# Save Training and Validation Loss/Accuracy Graphs
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(test_metrics_dir, 'train_val_accuracy.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(test_metrics_dir, 'train_val_loss.png'))
plt.close()

# Evaluate on Test Data
results = model.evaluate(test_data, verbose=1)
test_loss, test_accuracy = results[0], results[1]

# Save Test Results
with open(os.path.join(test_metrics_dir, 'test_results.txt'), 'w') as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_accuracy}\n")

# Predictions
true_labels = test_data.classes
predictions = model.predict(test_data, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_data.class_indices.keys(),
            yticklabels=test_data.class_indices.keys())
plt.title('Confusion Matrix (Test)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(test_metrics_dir, 'confusion_matrix_test.png'))
plt.close()

# Save Confusion Matrix as CSV
cm_df = pd.DataFrame(cm, index=test_data.class_indices.keys(),
                     columns=test_data.class_indices.keys())
cm_df.to_csv(os.path.join(test_metrics_dir, 'confusion_matrix_test.csv'))

# Classification Report with zero_division set to 1
report = classification_report(
    true_labels, predicted_labels, 
    target_names=test_data.class_indices.keys(),
    zero_division=1,  # Change to 0 if needed
    output_dict=True
)

# Convert report to DataFrame
report_df = pd.DataFrame(report).transpose()

# Check if expected metrics are present
metrics = ['precision', 'recall', 'f1-score']
if all(metric in report_df.index for metric in metrics):
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.bar(report_df.index[:-3], report_df[metric][:-3], color='blue')
        plt.title(f'{metric.capitalize()} per Class')
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45, ha='right')
        plt.savefig(os.path.join(test_metrics_dir, f'{metric}_per_class.png'))
        plt.close()
else:
    print("Not all expected metrics are present in the classification report.")

# Test Loss and Accuracy Graph
plt.figure(figsize=(6, 6))
metrics = ['Loss', 'Accuracy']
values = [test_loss, test_accuracy]
plt.bar(metrics, values, color=['red', 'blue'])
plt.title('Test Metrics')
plt.ylabel('Value')
plt.ylim(0, 1.2)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.savefig(os.path.join(test_metrics_dir, 'test_metrics_graph.png'))
plt.close()

print("All graphs and metrics saved successfully!")
