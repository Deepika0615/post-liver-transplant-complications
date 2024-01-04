# train test split chesina tarvata

# Count the occurrences of each class


class_counts = liverT['Complications'].value_counts()

# Calculate the proportion of each class
class_proportions = class_counts / len(liverT)

# Determine the threshold for imbalance
threshold = 0.1  # You can adjust this threshold based on your requirements

# Check if the data is balanced or imbalanced
is_balanced = all(prop >= threshold for prop in class_proportions)

if is_balanced:
    print("The data is balanced.")
else:
    print("The data is imbalanced.")
    
# Count class labels before SMOTE
target_counts_before = liverT['Complications'].value_counts()
class_labels_before = target_counts_before.index.tolist()

# Counting Class Labels before SMOTE
print("Class Counts (before SMOTE):")
print(target_counts_before)    
