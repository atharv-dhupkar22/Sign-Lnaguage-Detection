Real-Time Hand Sign Language to Text Translator
This project provides a real-time solution for translating hand sign language into text. By using a pre-trained neural network model, it detects and interprets hand signs (A-Z) from video input, displaying recognized characters as text on-screen.

Key Features
Hand Sign Recognition (A-Z): Detects and classifies hand signs via webcam.
Dynamic Sentence Formation: Continuously updates and displays a sentence based on recognized signs.
Space Detection: Adds a space when no hand sign is detected, enabling natural sentence flow.
Erase Feature: Press E to clear the current sentence entirely.
Usage Instructions
Run the Python script to initiate the webcam and hand sign detection.
Show hand signs in front of the camera for real-time recognition.
Recognized signs appear as text on the screen, forming a sentence.
Press E at any time to erase the current sentence.
Requirements
Python 3.x
TensorFlow
OpenCV
MediaPipe
Installation and Setup
Clone the repository:
bash
Copy code
git clone https://github.com/Amvic7/Real-time-Hand-Sign-Recognition.git
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the script:
bash
Copy code
python sign_language_recognition.py
Dataset and Model
The dataset consists of custom images captured using MediaPipe.
The neural network model (hand_sign_model.keras) is trained using Keras, capable of recognizing hand signs A-Z.
Contributors
Amvic
Atharv Dhupkar
Atharva Bhosale
Gauresh Bhoir
License
This project is licensed under the MIT License.
