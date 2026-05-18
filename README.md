# 🎙️ Speaker Recognition using CNN + Mel Spectrograms

A deep learning project that identifies **who is speaking** from a short audio clip. It converts raw `.wav` audio into Mel spectrograms and trains a Convolutional Neural Network (CNN) to classify the speaker's identity.

The model is trained to recognize 10 prominent public figures across politics, business, and entertainment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Speakers / Classes](#speakers--classes)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
- [Dataset Preparation](#dataset-preparation)

---

## Overview

This project tackles **closed-set speaker recognition** — given an audio clip, predict which of the known speakers is talking. The pipeline:

1. Load `.mp3` recordings and slice them into 3-second `.wav` clips
2. Convert each clip to a **Mel spectrogram** (a 2D frequency-time image)
3. Train a **CNN** on these spectrogram images
4. Predict the speaker for new unseen audio clips

---

## Speakers / Classes

The model is trained to identify the following 10 speakers:

| # | Speaker |
|---|---------|
| 1 | Mr. Narayan Murthy |
| 2 | Mr. Narendra Modi |
| 3 | Mr. Ratan Tata |
| 4 | Mr. S. Jaishankar |
| 5 | Mr. Sam Altman |
| 6 | Mr. Shah Rukh Khan |
| 7 | Mr. Shashi Tharoor |
| 8 | Mr. Steve Jobs |
| 9 | Mr. Vir Das |
| 10 | Mr. Warren Buffett |

---

## Project Structure

```
speech_recognitionproj/
│
├── program.py            # Main training + prediction script
├── check.py              # Alternate/validation training script (same pipeline)
├── speech_to_clip.py     # Utility: splits an MP3 into 3-second WAV clips
├── .gitignore
│
└── voice_data/           # Dataset directory (not included in repo)
    ├── Mr.Narayan Murthy/
    │   ├── clip_1.wav
    │   └── ...
    ├── Mr.Narendra Modi/
    └── ... (one folder per speaker)
```

---

## How It Works

### 1. Audio Preprocessing (`speech_to_clip.py`)
Raw MP3 recordings are split into 3-second WAV clips using `pydub`. Each clip becomes one training sample.

```python
clip_duration = 3 * 1000  # 3 seconds in milliseconds
clip.export(output_file_path, format='wav')
```

### 2. Mel Spectrogram Extraction
Each `.wav` clip is loaded with `librosa` and converted to a **128-band Mel spectrogram** normalized to decibels. All spectrograms are padded or truncated to a fixed width of 128 time frames, producing a consistent `(128, 128, 1)` input shape.

### 3. Model Training
The CNN is trained with:
- **Adam optimizer** (lr = 0.001)
- **ReduceLROnPlateau** scheduler (halves LR on stagnant val loss, patience = 3)
- **L2 regularization** + Dropout at every layer to reduce overfitting
- **40 epochs**, batch size 64, 70/30 train/test split

### 4. Prediction
New unseen WAV files are preprocessed identically, then passed through the trained model. The predicted class label is the speaker name.

---

## Model Architecture

```
Input: (128, 128, 1)  ← Mel spectrogram

Conv2D(32, 3×3, relu) + L2
MaxPooling2D(2×2)
Dropout(0.3)

Conv2D(64, 3×3, relu) + L2
MaxPooling2D(2×2)
Dropout(0.3)

Conv2D(128, 3×3, relu) + L2
MaxPooling2D(2×2)
Dropout(0.4)

Flatten
Dense(128, relu) + L2
Dropout(0.5)

Dense(num_classes, softmax)  ← Output: speaker probabilities
```

Loss: `sparse_categorical_crossentropy` | Metric: `accuracy`

---

## Requirements

Install dependencies with pip:

```bash
pip install numpy librosa keras tensorflow scikit-learn pydub
```

| Library | Purpose |
|---------|---------|
| `librosa` | Audio loading & Mel spectrogram extraction |
| `keras` / `tensorflow` | CNN model building and training |
| `numpy` | Array operations |
| `scikit-learn` | Train/test split |
| `pydub` | MP3 → WAV clip splitting |

> **Note:** `pydub` requires `ffmpeg` to be installed on your system.
> - **macOS:** `brew install ffmpeg`
> - **Ubuntu:** `sudo apt install ffmpeg`
> - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)

---

## Setup & Usage

### Step 1 — Prepare the dataset

Place your raw MP3 files inside `voice_data/<SpeakerName>/`. Then run the clip splitter to generate WAV clips:

```bash
python speech_to_clip.py
```

Edit the `mp3_file_path` and `output_folder` variables at the top of `speech_to_clip.py` for each speaker.

### Step 2 — Train the model

```bash
python program.py
```

This will:
- Load all WAV clips from `voice_data/`
- Extract Mel spectrograms
- Train the CNN for 40 epochs
- Print test loss and accuracy
- Run predictions on the predefined unseen clips

### Step 3 — Check predictions

At the end of `program.py`, a list of held-out clips is run through the model and the predicted speaker is printed:

```
The predicted class for voice_data/Mr.Steve Jobs/clip_126.wav is: Mr.Steve Jobs
```

---

## Dataset Preparation

The `voice_data/` directory should follow this structure:

```
voice_data/
└── <SpeakerName>/
    ├── clip_1.wav
    ├── clip_2.wav
    └── ...
```

Each subfolder name becomes the class label. The model automatically discovers all subfolders as classes — so adding a new speaker is as simple as creating a new folder with clips.

> The `voice_data/` directory **is committed to the repository** (see `.gitignore` — it is explicitly not excluded). Make sure clip files are present before running training.

---

## Notes

- Audio clips are capped at **3 seconds** during loading. Shorter clips are padded; longer ones are truncated.
- Trained model weights are **not saved** in the current scripts. To persist a trained model, add `model.save('model.h5')` after training.
- `check.py` is functionally identical to `program.py` and can be used for validation runs or experimentation.
