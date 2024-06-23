import pandas as pd

# Load the dataset
data = pd.read_csv('dataset.csv')

# Check the distribution of classes
class_distribution = data['Class'].value_counts()
print("Class distribution before sampling:")
print(class_distribution)

# Ensure there are enough samples of each class
num_class_0 = len(data[data['Class'] == 0])
num_class_1 = len(data[data['Class'] == 1])

if num_class_0 < 450 or num_class_1 < 450:
    raise ValueError("Dataset does not contain enough samples of each class for the requested sampling.")

# Sample 450 data points for each class
class_0_samples = data[data['Class'] == 0].sample(n=450, random_state=42)
class_1_samples = data[data['Class'] == 1].sample(n=450, random_state=42)

# Concatenate the samples to create the new dataset
balanced_data = pd.concat([class_0_samples, class_1_samples], axis=0).reset_index(drop=True)

# Shuffle the dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the distribution of classes in the new dataset
balanced_class_distribution = balanced_data['Class'].value_counts()
print("Class distribution after sampling:")
print(balanced_class_distribution)

# Save the new dataset to a CSV file
balanced_data.to_csv('dataset.csv', index=False)
print("Balanced dataset saved to 'dataset.csv'")
