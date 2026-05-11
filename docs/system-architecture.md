# SAMVAAD System Architecture

This diagram reflects the current code structure in `templates/app.py`, `sign_recog.py`, `samvaad_braille.py`, and `templates/avatar.js`.

```mermaid
flowchart LR
    user[User]

    subgraph browser[Browser Client]
        landing[Landing / Mode Pages<br/>index.html, sign.html, braille.html, learn.html]
        cam[Camera + Image Upload]
        speech[Web Speech APIs<br/>speech-to-text / text-to-speech]
        mp[Browser MediaPipe Hand Tracking]
        avatar[3D Avatar Player<br/>avatar.js + Three.js]
        recorder[Gesture Recorder / Learn Mode]
    end

    subgraph backend[Flask Backend]
        routes[Route Layer<br/>templates/app.py]
        signapi[/POST /process_landmarks/]
        sampleapi[/GET+POST gesture sample APIs/]
        learnapi[/GET common gestures API/]
        brailleapi[/POST /api/recognize-braille/]
        stable[Per-client Stable Gesture State]
    end

    subgraph signengine[Sign Recognition Engine]
        prep[Landmark Preparation<br/>mirror + normalize]
        rules[Rule-based Classifiers<br/>classify_right_hand / classify_left_hand]
        matcher[Recorded Sample Matcher<br/>JSONL embedding distance check]
    end

    subgraph brailleengine[Braille Recognition Engine]
        load[Image Decode]
        preproc[Preprocess<br/>grayscale, CLAHE, blur, adaptive threshold]
        dots[Dot Detection + Filtering]
        cells[Cell Segmentation]
        decode[Pattern Decode + Auto-spacing]
    end

    subgraph data[Local Project Assets]
        samples[dataset/gesture_samples/*.jsonl]
        brailleds[dataset/*.png]
        anims[templates/animations/*.fbx, *.glb]
        libs[templates/libs/*]
    end

    user --> landing
    user --> cam
    user --> speech

    landing --> avatar
    libs --> avatar
    anims --> avatar

    cam --> mp
    mp -->|21-point hand landmarks| signapi
    recorder -->|save labelled landmarks| sampleapi
    landing -->|learn gesture list| learnapi
    cam -->|braille image upload| brailleapi

    signapi --> routes
    sampleapi --> routes
    learnapi --> routes
    brailleapi --> routes

    routes --> prep
    prep --> rules
    prep --> matcher
    matcher --> samples
    routes --> stable
    rules --> stable
    matcher --> stable
    stable -->|recognized sign token| landing

    routes --> load
    load --> preproc
    preproc --> dots
    dots --> cells
    cells --> decode
    brailleds --> decode
    decode -->|text + confidence + metadata| landing

    landing -->|text to sign playback| avatar
    speech -->|recognized or spoken text| landing
    landing -->|speak final text| speech
    sampleapi --> samples
    learnapi --> anims
```

## Reading The Diagram

- The browser handles camera access, MediaPipe landmark extraction, speech input/output, and 3D sign playback.
- Flask serves both the static UI files and the JSON APIs used by sign recognition, Braille recognition, learn mode, and gesture sample recording.
- Sign recognition is hybrid: browser-side landmark capture plus server-side rule-based classification, recorded-sample matching, and temporal stabilization.
- Braille recognition is fully server-side and runs as an image-processing pipeline from upload to decoded text.
- Local assets power both inference and presentation: gesture samples in `dataset/gesture_samples`, Braille examples in `dataset`, and avatar animations in `templates/animations`.
