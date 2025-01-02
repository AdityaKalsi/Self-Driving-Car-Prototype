import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from subprocess import call


class SteeringAnglePredictor:
    def __init__(self, model_path):
        # Load the saved model using compatible method
        try:
            # Use TFSMLayer if the model is in TensorFlow SavedModel format
            self.model = TFSMLayer(model_path, call_endpoint="serving_default")
        except ValueError as e:
            print(f"Error loading model: {e}")
            print("Ensure the model is in .keras, .h5, or compatible format.")
            raise
        self.smoothed_angle = 0

    def predict_angle(self, image):
        # Predict the steering angle from the image
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        prediction = self.model(image, training=False)
        degrees = prediction.numpy()[0][0] * 180.0 / 3.14159265  # Convert to degrees
        return degrees

    def smooth_angle(self, predicted_angle):
        if self.smoothed_angle == 0:
            self.smoothed_angle = predicted_angle
        else:
            self.smoothed_angle += 0.2 * pow(abs(predicted_angle - self.smoothed_angle), 2.0 / 3.0) * (
                    predicted_angle - self.smoothed_angle) / abs(predicted_angle - self.smoothed_angle)
        return self.smoothed_angle


class DrivingSimulator:
    def __init__(self, predictor, data_dir, steering_image_path, is_windows=False):
        self.predictor = predictor
        self.data_dir = data_dir
        self.steering_image = self.load_steering_image(steering_image_path)
        self.is_windows = is_windows
        self.rows, self.cols = self.steering_image.shape

    @staticmethod
    def load_steering_image(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Steering wheel image not found: {path}")
        return cv2.imread(path, 0)

    def start_simulation(self):
        i = 0
        while cv2.waitKey(10) != ord('q'):
            try:
                image_path = os.path.join(self.data_dir, f"{i}.jpg")
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    break

                full_image = cv2.imread(image_path)
                resized_image = cv2.resize(full_image[-150:], (200, 66)) / 255.0

                predicted_angle = self.predictor.predict_angle(resized_image)
                smoothed_angle = self.predictor.smooth_angle(predicted_angle)

                if not self.is_windows:
                    call("clear")
                print(f"Predicted steering angle: {predicted_angle:.2f} degrees")

                self.display_frames(full_image, smoothed_angle)
                i += 1

            except Exception as e:
                print(f"Error during simulation: {e}")
                break

        cv2.destroyAllWindows()

    def display_frames(self, full_image, smoothed_angle):
        cv2.imshow("frame", full_image)
        rotation_matrix = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -smoothed_angle, 1)
        rotated_steering_wheel = cv2.warpAffine(self.steering_image, rotation_matrix, (self.cols, self.rows))
        cv2.imshow("steering wheel", rotated_steering_wheel)


if __name__ == "__main__":
    # Define paths
    model_path = "saved_models/regression_model/model"  # Ensure correct format for compatibility
    data_dir = "data/driving_dataset"
    steering_image_path = "data/steering_wheel_image.jpg"

    # Determine if running on Windows
    is_windows = os.name == 'nt'

    try:
        # Initialize predictor and simulator
        predictor = SteeringAnglePredictor(model_path)
        simulator = DrivingSimulator(predictor, data_dir, steering_image_path, is_windows)

        # Start simulation
        simulator.start_simulation()

    except FileNotFoundError as fnf_error:
        print(f"File error: {fnf_error}")
    except Exception as e:
        print(f"Unexpected error: {e}")
