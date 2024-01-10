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


# Define a learning rate schedule using LearningRateScheduler
def lr_schedule(epoch, min_lr, max_lr, cycle_length):
    cycle = 1 + epoch // (2 * cycle_length)
    x = abs(epoch / cycle_length - 2 * cycle + 1)
    lr = min_lr + (max_lr - min_lr) * max(0, 1 - x)
    return lr


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

# Define the number of epochs and cycle length
epochs = 10
cycle_length = 5

# Define the minimum and maximum boundary values for learning rate
min_lr = 0.001
max_lr = 0.01

# Train the model using training data with validation and save the accuracy history as a variable
history = []
for epoch in range(epochs):
    lr = lr_schedule(epoch, min_lr, max_lr, cycle_length)
    optimizer.lr.assign(lr)
    history.append(mnist_model.fit(train_images, train_labels, steps_per_epoch=10,
                                   validation_split=0.2, verbose=2))

# Evaluate the trained model using test data and print the test accuracy
test_loss, test_acc = mnist_model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Create a plot for visualization
plt.figure(figsize=(8, 5))

# Plot the learning rate curve multiplied by 50 for better visualization on the graph
lr_values = [lr_schedule(epoch, min_lr, max_lr, cycle_length) * 50 for epoch in range(epochs)]
plt.plot(lr_values, label='Learning Rate', linestyle='dashed')

# Plot the training accuracy curve
plt.plot([history[i].history['sparse_categorical_accuracy'][-1] for i in range(epochs)], label='Training Accuracy')

# Plot the validation accuracy curve
plt.plot([history[i].history['val_sparse_categorical_accuracy'][-1] for i in range(epochs)],
         label='Validation Accuracy')

# Set labels and title for the plot
plt.xlabel('Epoch')
plt.title('Model Curves')

# Display the legend and ensure plot layout
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()