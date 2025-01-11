# Malicious-Object-Detection
Developed a malicious object detection system using YOLOv8 for initial detection. Refined results by analyzing image depth with Depth_Anything_v2 to verify object presence. Finally, classified detected objects as real or fake using a YOLO-based classifier, ensuring accurate and reliable detection.

**Malicious Object Detection System**  

This project focuses on the detection and classification of malicious objects in images, employing advanced deep learning techniques. It is designed to ensure high precision and reliability by leveraging multiple stages of object detection, depth analysis, and classification.  

### Repository Structure:  
1. **`final.ipynb`**: Contains the code explanation and step-by-step implementation. This notebook provides a detailed walkthrough of the methods used, the model pipeline, and insights into the decision-making process.  
2. **`oneShot.py`**: A standalone Python script that runs the entire model pipeline in one go. Execute this file to directly obtain results for object detection and classification.  
3. **`our_images/`**: Directory containing the images used for testing and evaluation.

### Dataset Details:  
The model was trained on a curated dataset of 11,000 images sourced from [Roboflow](https://roboflow.com/). The dataset comprises six malicious object classes:  
- **Gun**  
- **Smoke**  
- **Blade**  
- **Grenade**  
- **Fire**  
- **Axe**  

Careful preprocessing ensured equal distribution across these classes, providing a balanced training set for optimal performance.  

### Model Pipeline:  
1. **Object Detection**:  
   The first stage utilizes **YOLOv8**, a cutting-edge object detection model known for its speed and accuracy. It identifies potential malicious objects in the input images.  

2. **Depth Analysis**:  
   Detected objects are refined by analyzing their depth using **Depth_Anything_v2**. This step verifies whether the detected object is genuinely present in the 3D space of the image, reducing false positives caused by flat or misleading regions.  

3. **Object Classification**:  
   Finally, a **YOLO-based classifier** determines whether the detected object is real or fake, ensuring the validity of the results.  

### Key Features:  
- **End-to-End System**: From detection to verification and classification, the pipeline is designed for seamless operation.  
- **Robust Dataset**: The use of a diverse and well-balanced dataset ensures the model's generalizability across real-world scenarios.  
- **High Precision**: By incorporating depth analysis, the system minimizes errors, achieving greater reliability.  

This project demonstrates the practical application of deep learning in safety-critical scenarios, making it a valuable tool for malicious object detection and prevention. The modular design also allows easy adaptation to other use cases involving object detection and classification.  

For a detailed understanding, refer to `final.ipynb`, or run `oneShot.py` for quick results. We welcome feedback and contributions to enhance this system further." 
