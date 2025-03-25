# Face Mood Detection

A real-time facial expression recognition application capable of detecting emotions from webcam input. The application uses OpenCV for face detection and features a dual-approach to emotion recognition:

1. Machine Learning-based detection using ONNX models (when available)
2. Rule-based detection using traditional computer vision techniques

## Features

- Real-time face detection
- Emotion recognition (Happy, Sad, Angry, Neutral, Surprised, etc.)
- Confidence scoring for detected emotions
- Calibration capability to personalize emotion detection
- Debug visualization and logging

## Requirements

- OpenCV 4.x
- CMake 3.x
- C++11 or higher compiler
- Webcam

## Building the Application

```bash
mkdir build
cd build
cmake ..
make
```

## Running the Application

```bash
./face_mood_detection
```

### Command-line Options

- `--debug`: Enable debug output and visualization
- `--cascade=path/to/classifier.xml`: Specify a custom face classifier
- `--model=path/to/model.onnx`: Specify a custom emotion recognition model
- `--no-ml`: Disable machine learning-based emotion detection

## Calibration

You can calibrate the system for better emotion recognition:
- Press '1' to calibrate Neutral expression
- Press '2' to calibrate Happy expression
- Press '3' to calibrate Sad expression
- Press '4' to calibrate Angry expression
- Press '5' to calibrate Surprised expression
- Press '0' to reset calibration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Face detection using OpenCV Haar Cascades
- Emotion recognition inspired by FER+ dataset 