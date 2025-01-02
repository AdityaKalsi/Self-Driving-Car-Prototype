# **Overview**
<p>&nbsp;</p>
This project aims to develop a self-driving car system using advanced computer vision and machine learning techniques. The solution is divided into two distinct pipelines:

- **Pipeline 1**: Focuses on lane detection, object detection (cars and traffic signs), and steering angle prediction to ensure safe navigation.
- **Pipeline 2**: Dedicated to pothole prediction, which enhances road safety by detecting and avoiding damaged road sections.

The project leverages cutting-edge architectures, such as CNN-based models inspired by the NVIDIA paper for steering angle prediction and YOLOv11 for object detection, to create a robust, real-time autonomous driving solution.

---

## **Features**

### **Pipeline 1: Navigation and Safety**
- **Lane Detection**: Detects lane boundaries using computer vision techniques to ensure the car stays within the lane.
- **Object Detection**: Identifies nearby cars and traffic signs to make informed decisions.
- **Steering Angle Prediction**: Using a deep learning model to predict the optimal steering angle for smooth and safe driving.

### **Pipeline 2: Road Safety**
- **Pothole Detection**: Identifies and classifies potholes in real-time to avoid potential road hazards.

### **General Features**
- **Real-Time Processing**: Ensures decisions are made instantly to handle dynamic road scenarios.
- **Integrated Vision Models**: Combines vision-based techniques for accurate environmental understanding.
- **Scalable Architecture**: The modular design allows for easy enhancements and integration of new features.

---

## **Step-by-Step Process**

### **Pipeline 1**
1. **Input Video Stream**: The system receives live video input from a front-mounted camera.
2. **Lane Detection**:
   - Preprocesses the input using techniques like Gaussian blur and Canny edge detection.
   - Applies Hough Transform or polynomial fitting to detect lanes.
3. **Object Detection**:
   - YOLOv11 processes frames to identify objects (vehicles, traffic signs, etc.).
   - Outputs bounding boxes and class labels for detected objects.
4. **Steering Angle Prediction**:
   - A CNN inspired by NVIDIA's architecture predicts the steering angle based on the processed input.
   - Adjusts the car's steering system in real-time.

### **Pipeline 2**
1. **Input Pothole Data**: Processes road surface information from an additional downward-facing camera or LiDAR.
2. **Feature Extraction**: Uses image processing and machine learning to detect irregularities on the road.
3. **Classification**: Identifies potholes and provides their location for avoidance.

---

## **Unique Idea Brief**

The project addresses both primary aspects of autonomous driving: **Navigation** and **Road Safety**.

- **Navigation System**: Focuses on seamless lane detection, object identification, and precise steering control. This ensures that the car navigates safely through various road conditions and traffic scenarios.
- **Safety Mechanism**: Enhances the system by detecting potholes, a critical real-world challenge, enabling smoother and safer rides.

---

## **Highlights**

- **NVIDIA-Inspired CNN**: The model leverages a specialized convolutional neural network for steering angle prediction, ensuring reliable navigation decisions.
- **YOLOv11 for Object Detection**: Offers high accuracy and real-time detection of vehicles and traffic signs.
- **Pothole Prediction Module**: A dedicated pipeline to detect and classify road damage, setting it apart from standard self-driving systems.

---

## **Tools and Technologies**

- **Lane Detection**: OpenCV for edge detection and image preprocessing.
- **Object Detection**: YOLOv11 for real-time object detection.
- **Steering Angle Prediction**: Custom CNN inspired by NVIDIA's architecture.
- **Pothole Detection**: Image processing and machine learning-based classification.

---

## **Dataset Sources**

- Custom road videos and datasets for lane detection.
- Open-source datasets for object detection and pothole identification.

