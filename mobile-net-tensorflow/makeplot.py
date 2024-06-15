import matplotlib.pyplot as plt

# Define the data for the epochs
epochs = list(range(1, 11))
average_losses = [4.0141, 3.3338, 3.0673, 2.9131, 2.7927, 2.7115, 2.6426, 2.5881, 2.5542, 2.5155]
validation_accuracies = [0.1618, 0.1810, 0.1835, 0.1842, 0.1988, 0.2010, 0.2046, 0.2046, 0.2058, 0.2142]

# Plotting validation accuracy and average accuracy
plt.figure(figsize=(14, 5))

# Validation Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, validation_accuracies, marker='o', color='blue', label='Validation Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.legend()

# Average Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, average_losses, marker='o', color='red', label='Average Loss')
plt.title('Average Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
