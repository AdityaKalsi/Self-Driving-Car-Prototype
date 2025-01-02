import tensorflow as tf

def weight_variable(shape):
    initial = tf.random.normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

# Define the model using the Keras functional API or Sequential
class SteeringModel(tf.keras.Model):
    def __init__(self):
        super(SteeringModel, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(24, (5, 5), strides=2, activation='relu', padding='valid')
        self.conv2 = tf.keras.layers.Conv2D(36, (5, 5), strides=2, activation='relu', padding='valid')
        self.conv3 = tf.keras.layers.Conv2D(48, (5, 5), strides=2, activation='relu', padding='valid')
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid')
        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid')
        
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1164, activation='relu')
        self.fc2 = tf.keras.layers.Dense(100, activation='relu')
        self.fc3 = tf.keras.layers.Dense(50, activation='relu')
        self.fc4 = tf.keras.layers.Dense(10, activation='relu')
        self.fc5 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        
        # Return scaled atan output
        return tf.atan(x) * 2

# Example input data
x_input = tf.random.normal([1, 66, 200, 3])  # Simulate a single image

# Create model instance
model = SteeringModel()

# Get the prediction (assuming a batch size of 1)
predicted_angle = model(x_input)
print(f"Predicted steering angle: {predicted_angle.numpy()[0]}")


