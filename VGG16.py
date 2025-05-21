import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ====================
# Setup directories
# ====================
train_dir = r'D:\MLLLLLL\split3\train'
val_dir = r'D:\MLLLLLL\split3\val'
test_dir = r'D:\MLLLLLL\split3\test'
plot_dir = r'D:\MLLLLLL\Output4\vgg16_output'
os.makedirs(plot_dir, exist_ok=True)

# ====================
# Parameters
# ====================
img_size = (224, 224)
batch_size = 32
learning_rate = 0.001
epochs = 10

# ====================
# Data Preparation
# ====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True)

val_data = val_test_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

test_data = val_test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

# ====================
# Load Pretrained VGG16 Model
# ====================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(len(train_data.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ====================
# Compile & Train
# ====================
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop],
    verbose=1
)

# ====================
# Save the Model
# ====================
model_path = r'D:\MLLLLLL\Output4\vgg16_output\my_model.keras'
model.save(model_path)

# ====================
# Accuracy Curve
# ====================
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(plot_dir, "accuracy_curve.png"))
plt.close()

# ====================
# Loss Curve
# ====================
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
plt.close()

# ====================
# Evaluate on Test Data
# ====================
test_loss, test_acc = model.evaluate(test_data, verbose=1)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# ====================
# Confusion Matrix
# ====================
y_true = test_data.classes
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)
class_labels = list(test_data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
plt.close()

# ====================
# Classification Report (CSV)
# ====================
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(plot_dir, "classification_report.csv"))

# ====================
# Optional Fine-Tuning (Uncomment to use after initial training)
# ====================
# base_model.trainable = True
# for layer in base_model.layers[:-4]:
#     layer.trainable = False
# model.compile(optimizer=Adam(learning_rate=1e-5),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# history_finetune = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=10,
#     callbacks=[early_stop],
#     verbose=1
# )
