# Speech Emotion Recognition

Classifies emotions in speech audio using **MFCC features** and a **CNN**
trained on the [RAVDESS](https://zenodo.org/record/1188976) dataset.

Supports 8 emotion classes:
`neutral` · `calm` · `happy` · `sad` · `angry` · `fearful` · `disgust` · `surprised`

---

## Project Structure

```
speech-emotion-recognition/
├── data/
│   ├── raw/                   ← Place RAVDESS Actor_XX/ folders here
│   └── processed/             ← Auto-generated: .npy feature arrays
├── models/
│   └── saved_model/           ← Auto-generated: model + label encoder
├── notebooks/                 ← Optional: Jupyter exploration notebooks
├── outputs/
│   └── plots/                 ← Auto-generated: training & confusion plots
├── src/
│   ├── __init__.py
│   ├── config.py              ← All paths, constants, hyperparameters
│   ├── data_loader.py         ← Discovers files, parses emotion labels
│   ├── feature_extraction.py  ← MFCC extraction + CNN reshaping
│   ├── preprocess.py          ← Full preprocessing pipeline + loaders
│   ├── model.py               ← CNN architecture
│   ├── train.py               ← Training loop + callbacks + plots
│   ├── evaluate.py            ← Test-set metrics + confusion matrix
│   └── predict.py             ← Inference on a single audio file
├── main.py                    ← CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the RAVDESS dataset

Download **Audio_Speech_Actors_01-24** from
[https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
and extract so your layout looks like:

```
data/raw/Actor_01/03-01-01-01-01-01-01.wav
data/raw/Actor_01/03-01-01-01-01-02-01.wav
data/raw/Actor_02/...
...
data/raw/Actor_24/...
```

---

## Running the Pipeline

### Full pipeline (one command)
```bash
python main.py --stage all
```

### Or step by step

**Step 1 — Preprocess** *(run once; takes a few minutes)*
```bash
python main.py --stage preprocess
```
Extracts 40-coefficient MFCCs for all 1 440 clips, encodes labels, saves
train/test arrays and the `LabelEncoder` to disk.

**Step 2 — Train**
```bash
python main.py --stage train
```
Trains the CNN with early stopping. Saves the best model and a
training-history plot to `outputs/plots/training_history.png`.

**Step 3 — Evaluate**
```bash
python main.py --stage evaluate
```
Prints test accuracy, a per-class classification report, and saves a
confusion matrix to `outputs/plots/confusion_matrix.png`.

**Step 4 — Predict on a new file**
```bash
python main.py --stage predict --file path/to/your_audio.wav
```

---

## Expected Results

| Metric | Typical value |
|---|---|
| Test accuracy | 55 – 70 % |
| Best classes | angry, happy, surprised |
| Hardest classes | neutral, calm (acoustically similar) |

*Accuracy depends on random seed and hardware. RAVDESS + simple CNN is
intentionally baseline; accuracy can be improved with data augmentation
or an LSTM-based model.*

---

## Troubleshooting

| Error | Fix |
|---|---|
| `FileNotFoundError: Raw data directory not found` | Extract RAVDESS into `data/raw/` so `Actor_01/` … `Actor_24/` are direct children |
| `Processed data not found` | Run `python main.py --stage preprocess` first |
| `Trained model not found` | Run `python main.py --stage train` first |
| `No module named 'librosa'` | Run `pip install -r requirements.txt` |
