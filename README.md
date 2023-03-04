# FaceMesh Eye Region Extractor

This Python project uses the FaceMesh module from Mediapipe to extract the eye region from a video stream or image. The eye region is then displayed in a separate window, along with the full image.

## What is Mediapipe?

Mediapipe is an open-source framework developed by Google that provides a collection of pre-built, customizable building blocks for building machine learning (ML) and computer vision (CV) pipelines. Mediapipe is designed to simplify the development process for ML and CV applications, and it includes a wide range of pre-built models and tools that can be used to build and deploy ML and CV pipelines for a variety of applications.

## FaceMesh Landmarks

The FaceMesh module in Mediapipe extracts 468 landmarks from the face, including 168 landmarks for the eyes, eyebrows, and forehead. To visualize the locations of these landmarks, you can refer to the canonical face model visualization provided by Mediapipe:

![Canonical Face Model Visualization](https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

## Installation

1. Clone the repository or download the source code.
2. Open a terminal or command prompt in the project directory.
3. Run the command `pip install -r requirements.txt` to install the required packages.
4. Run the `runner/main.py` script to extract the eye region from a video stream or image.
