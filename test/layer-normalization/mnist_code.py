import tensorflow as tf
import matplotlib.pyplot as plt


# Define the model class for MNIST classification
class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)


# Define optimizer, loss, and metrics for model compilation
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

# Create an instance of the MNISTModel class
mnist_model = MNISTModel()

# Compile the model using specified optimizer, loss, and metrics
mnist_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Load and preprocess the MNIST dataset for training and testing
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Train the model using training data with validation and save the accuracy history as a variable
history = mnist_model.fit(train_images, train_labels, epochs=10, steps_per_epoch=10,
                          validation_split=0.2, verbose=2)

# Evaluate the trained model using test data and print the test accuracy
test_loss, test_acc = mnist_model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Create a plot for visualization
plt.figure(figsize=(8, 5))

# Plot the training accuracy curve
plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')

# Plot the validation accuracy curve
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')

# Set labels and title for the plot
plt.xlabel('Epoch')
plt.title('Model Curves')

# Display the legend and ensure plot layout
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()