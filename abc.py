import numpy as np
import pandas as pd

# Define the parameters
num_classes = 11
images_per_class = 628
total_images = num_classes * images_per_class
accuracy = 97.84 / 100

# Calculate total correct predictions
total_correct = int(total_images * accuracy)
total_incorrect = total_images - total_correct

# Initialize the confusion matrix
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

# Fill the diagonal with correct predictions
correct_per_class = total_correct // num_classes
remaining_correct = total_correct % num_classes

for i in range(num_classes):
    confusion_matrix[i, i] = correct_per_class + (1 if i < remaining_correct else 0)

# Distribute the incorrect predictions across off-diagonal elements
errors_remaining = total_incorrect
while errors_remaining > 0:
    for i in range(num_classes):
        if errors_remaining == 0:
            break
        for j in range(num_classes):
            if i != j and errors_remaining > 0:
                confusion_matrix[i, j] += 1
                errors_remaining -= 1

# Convert to a pandas DataFrame for better visualization
class_names = [
    "Cercospora", "Curl", "Flea Beetles", "Hadda Beetles", "Healthy",
    "LeafhopperJassids", "Magnesium Deficiency", "Phomposist Blast",
    "TMV (Tobacco Mosaic Virus)", "Tobacco Caterpillar", "Verticillium Wilt"
]

confusion_df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_df)

# Verify the accuracy calculation
correct_predictions = np.trace(confusion_matrix)
calculated_accuracy = correct_predictions / total_images * 100

print(f"\nTotal Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Calculated Accuracy: {calculated_accuracy:.2f}%")