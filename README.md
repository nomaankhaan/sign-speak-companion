# Sign and Speak Companion

This repository contains all the base code required for data gathering, model training, and classification of sign language gestures based on ISL (Indian Sign Language) using Google's MediaPipe.


## Project Overview

A Deep Learning Neural Network is used to classify sign language gestures.
For data gathering purposes (increasing the number of static/dynamic gestures), Google MediaPipe is used to access the webcam via OpenCV, mapping the key points around each hand with 21 points and a center point on the nose to maintain reference.

## Clone the Repository

To clone this repository, follow these steps:

Create an empty project directory and run the following commands:

```bash
  git clone https://github.com/nomaankhaan/sign-speak-companion.git
  cd sign-speak-companion
```

## Steps for New Data Gathering

1. Add a new gesture label in key_classifier_label.csv.
2. Run the data_collector.py script and toggle to mode 0 by pressing the k key. Enter the specified row number for the newly added gesture in the CSV file to store the corresponding data.
3. Perform the gesture in front of the camera:
  a. Press s for single data capture.
  b. Press Enter for continuous data capture at each   specified   timestamp.

4. To see how many rows have been collected for a specified gesture, run insight.py.
5. Run the keypoint_classification_EN.ipynb notebook to retrain the neural network.

### Note

To add dynamic gestures, separately add both parts of the gesture to the CSV (e.g., hello_1, hello_2) and gather the data for each part separately. After data gathering, modify transition.py to ensure that hello is only detected after hello_2 is detected following hello_1.

### Steps to Classify a Newly Added or Existing Gesture

1. Run data_collector.py and toggle to mode 1 by pressing the k key to turn on Detecting mode.
2. Perform the gesture in front of the camera, and the results will be visible in the camera frame.

## Research Paper

[Research Link](https://www.ijirmf.com/wp-content/uploads/IJIRMF202405004-min.pdf)
