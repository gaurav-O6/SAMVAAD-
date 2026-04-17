# SAMVAAD

SAMVAAD is a multimodal assistive communication system that brings together sign language, text, speech, and Braille workflows in one project. It is built as an accessibility-focused final year project for bridging communication gaps across hearing, speech, and visual impairments.

## What It Does

- Real-time sign recognition from browser-captured hand landmarks
- Text-to-sign playback using a 3D avatar and sign animation library
- Voice input and text-to-speech in the web interface
- Braille image recognition with confidence scoring and debug overlays
- Gesture sample recording for extending or calibrating recognition

## Project Structure

```text
SAMVAAD/
+-- app.py                  # Root Flask entry point
+-- requirements.txt
+-- sign_recog.py           # Landmark-based sign classifier
+-- samvaad_braille.py      # Braille recognition engine + CLI
+-- dataset/                # Braille samples and gesture samples
+-- templates/
|   +-- app.py              # Flask app module
|   +-- avatar.js           # 3D avatar controller
|   +-- index.html          # Landing page
|   +-- sign.html           # Sign recognition mode
|   +-- braille.html        # Braille mode
|   +-- learn.html          # Learn/common gestures mode
|   +-- animations/         # FBX avatar + sign animations
|   +-- libs/               # Three.js / loader dependencies
+-- tests/                  # Lightweight regression tests
```

## Tech Stack

- Python
- Flask
- Flask-CORS
- MediaPipe
- OpenCV
- NumPy
- Browser Web Speech APIs
- Three.js

## Quick Start

1. Clone the repository and enter the project folder.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the backend from the project root:

```bash
python app.py
```

5. Open [http://localhost:5000](http://localhost:5000).

On Windows, you can also use [START_SAMVAAD.bat](</D:/SAMVAAD/START_SAMVAAD.bat>) after creating the virtual environment.

Recommended Windows flow:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Testing

Run the regression tests from the project root:

```bash
python -m unittest discover -s tests
```

If you use a virtual environment, run the tests from that same environment so Flask, Werkzeug, MediaPipe, and OpenCV are available.

You can also run the Braille recognizer directly:

```bash
python samvaad_braille.py --test dataset
```

## Notes

- The 3D avatar currently uses FBX assets from [templates/animations](</D:/SAMVAAD/templates/animations>).
- Sign recognition uses browser-side MediaPipe landmarks and server-side gesture stabilization.
- Gesture sample files are stored under `dataset/gesture_samples`.

## Roadmap Ideas

- Sentence-level sign translation
- Better evaluation metrics for gesture accuracy
- More curated Braille benchmark images
- Easier packaging for demos and classroom setup

## Author

Gaurav  
Final Year Project - SAMVAAD

## License

This project is intended for academic and research use.
