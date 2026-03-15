SAMVAAD – Multimodal Communication System

SAMVAAD is an AI-powered assistive communication platform designed to bridge communication gaps for individuals with hearing, speech, and visual impairments. The system integrates multiple technologies to enable seamless interaction through speech, sign language, and braille recognition.

The goal of SAMVAAD is to create a unified interface where different forms of human communication can be translated and understood across modalities.

Features
Speech to Text

Converts spoken language into readable text in real time.

Text to Speech

Allows written text to be converted into spoken output.

Sign Language Recognition

Uses computer vision to recognize hand gestures and translate them into text.

Braille Recognition

Detects braille patterns from images and converts them into readable characters.

Gesture and Motion Capture

Motion capture data is used to improve gesture analysis and dataset generation.

Motion Capture Integration

Gesture datasets used in this project were generated and analyzed using Marionette Studio, a motion capture software that records and tracks human movement.

Using Marionette mocap allowed us to:

Capture realistic human gesture movements

Improve training data for sign language recognition

Analyze motion patterns more accurately

Simulate gesture sequences for testing

This significantly improves gesture detection reliability compared to static image datasets.

Technologies Used

Python

Computer Vision

Machine Learning

Flask Web Framework

OpenCV

Speech Recognition Libraries

Motion Capture Data Processing

Project Structure
SAMVAAD
│
├── dataset
├── templates
├── static
├── app.py
├── mediapipe_test.py
├── requirements.txt
└── README.md
How to Run the Project
1 Clone the repository
git clone https://github.com/YOUR_USERNAME/SAMVAAD.git
2 Navigate to the project
cd SAMVAAD
3 Install dependencies
pip install -r requirements.txt
4 Run the application
python app.py
Future Improvements

Real-time sign language sentence translation

Integration with mobile platforms

Expanded braille dataset

Enhanced motion capture datasets for gesture learning

Author

Gaurav
Final Year Project – SAMVAAD

License

This project is developed for academic and research purposes.
