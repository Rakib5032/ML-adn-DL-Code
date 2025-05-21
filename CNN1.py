import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Suppress oneDNN messages (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set directories with raw strings
train_dir = r'D:\AI LAB\Face Dataset\train'
test_dir = r'D:\AI LAB\Face Dataset\test'
plot_dir = r'D:\AI LAB\plots'
os.makedirs(plot_dir, exist_ok=True)

# Step 1: Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,  # Fine for faces
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),    # FER-2013 standard size
    color_mode='grayscale',  # FER-2013 is grayscale
    batch_size=32,
    class_mode='categorical' # 7 classes
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

# Print class info
print(f"Number of classes: {train_generator.num_classes}")
print(f"Training samples: {train_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")

# Step 2: Build the CNN Model with Dropout in the Middle
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # 1 channel for grayscale
    MaxPooling2D((2, 2)),
    Dropout(0.2),  # Dropout after first pooling
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),  # Dropout after second pooling
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),  # Dropout after third pooling
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Keep 50% dropout in dense layer
    Dense(7, activation='softmax')  # 7 classes for FER-2013
])

# Step 3: Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Step 4: Train the Model with Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[early_stopping]
)

# Step 5: Evaluate
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 6: Plot Results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Facial Expression Recognition Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Save the plot to the specified directory
plot_path = os.path.join(plot_dir, 'face_expression_accuracy.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
plt.show()

# Step 7: Display Predictions
sample_images, sample_labels = next(test_generator)
predictions = model.predict(sample_images)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # FER-2013 classes

for i in range(5):
    plt.imshow(sample_images[i].reshape(48, 48), cmap='gray')  # Grayscale display
    pred_label = class_names[np.argmax(predictions[i])]
    true_label = class_names[np.argmax(sample_labels[i])]
    plt.title(f"Predicted: {pred_label}, Actual: {true_label}")
    plt.axis('off')
    plt.show()
