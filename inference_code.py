# -*- coding: utf-8 -*-
"""Inference code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/130yw1ciNoFxo-RHhhcXYcUQOQaUzE1xn
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !wget https://fisd-dataset.s3.amazonaws.com/fisd-asanti-twi-10p.zip
# !unzip /content/twi.zip
# !unzip /content/fisd-asanti-twi-10p.zip

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install librosa
# !pip install datasets
# !pip install jiwer
# !pip install evaluate

import evaluate

# Load the Word Error Rate (WER) metric
metric = evaluate.load("wer")

"""# Using the best model so far."""

!pip install librosa datasets transformers soundfile

import os
import re
import unicodedata
import string
import librosa
import torch
import pandas as pd
import evaluate
from datasets import Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# File paths for model , audio and csv transcriptions
model_checkpoint = "/content/drive/MyDrive/whisper-small-tw/checkpoint-1400"
test_csv_path = "/content/fisd-asanti-twi-10p/data.csv"
audio_base_path = "/content/fisd-asanti-twi-10p/audios"


# Load the processor and model from the checkpoint

processor = WhisperProcessor.from_pretrained(model_checkpoint)
model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load the test CSV into a pandas DataFrame
test_df = pd.read_csv(test_csv_path, delimiter="\t", on_bad_lines='skip')

# Ensure required columns exist. If your CSV has different column names, change these.

if "Audio Filepath" not in test_df.columns or "Transcription" not in test_df.columns:
    raise ValueError("The test CSV must have 'Audio Filepath' and 'Transcription' columns.")

# Update the audio file paths to full paths if needed
test_df["Audio Filepath"] = test_df["Audio Filepath"].apply(
    lambda x: os.path.join(audio_base_path, os.path.basename(x))
)


# Convert the pandas DataFrame to a Dataset

test_dataset = Dataset.from_pandas(test_df)


# Define a function to load and resample audio
def load_audio(example):
    waveform, sr = librosa.load(example["Audio Filepath"], sr=16000)
    example["audio"] = {
        "array": waveform,
        "sampling_rate": 16000
    }
    return example

# Apply the load_audio function to add "audio" column
test_dataset = test_dataset.map(load_audio, batched=False)


# Define a function to run inference on each sample

def map_to_pred(batch):
    # Process the audio input
    inputs = processor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"], return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            task="transcribe",
            use_cache=True,
            max_length=225
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    batch["prediction"] = transcription
    return batch


# Run inference on the test set

test_pred = test_dataset.map(map_to_pred, batched=False)


# Compute WER
metric = evaluate.load("wer")
test_wer = 100 * metric.compute(predictions=test_pred["prediction"], references=test_pred["Transcription"])
print("Test WER:", test_wer)



!pip install librosa datasets transformers soundfile

import numpy as np
import os
import re
import unicodedata
import string
import librosa
import torch
import pandas as pd
import evaluate
from datasets import Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_checkpoint = "/content/drive/MyDrive/please/checkpoint-1000"
test_csv_path = "/content/fisd-asanti-twi-10p/data.csv"
audio_base_path = "/content/fisd-asanti-twi-10p/audios"

# Load the processor and model from the checkpoint

processor = WhisperProcessor.from_pretrained(model_checkpoint)
model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the test CSV into a pandas DataFrame

import pandas as pd
test_df = pd.read_csv(test_csv_path, delimiter="\t", on_bad_lines='skip')

if "Audio Filepath" not in test_df.columns or "Transcription" not in test_df.columns:
    raise ValueError("The test CSV must have 'Audio Filepath' and 'Transcription' columns.")

# Fix the audio paths
import os
test_df["Audio Filepath"] = test_df["Audio Filepath"].apply(
    lambda x: os.path.join(audio_base_path, os.path.basename(x))
)

metric = evaluate.load("wer")



    # Decode prediction
    predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Print result immediately
    print(f"File {i+1}/{len(test_df)}: {audio_path}")
    print("Predicted:", predicted_text)
    print("Reference:", reference_text)
    print("-" * 40)

    # Store for WER calculation
    predictions.append(predicted_text)
    references.append(reference_text)


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    distance = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        distance[i][0] = i
    for j in range(n+1):
        distance[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            distance[i][j] = min(
                distance[i-1][j] + 1,
                distance[i][j-1] + 1,
                distance[i-1][j-1] + cost
            )
    return distance[m][n]

def compute_character_error_rate(predictions, references):
    total_distance = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        pred = pred.strip()
        ref = ref.strip()
        dist = levenshtein_distance(ref, pred)
        total_distance += dist
        total_chars += len(ref)
    if total_chars == 0:
        return 0.0
    return total_distance / total_chars

predictions = []
references = []

for i, row in test_df.iterrows():
    audio_path = row["Audio Filepath"]
    reference_text = row["Transcription"]

    # Load and resample audio
    waveform, sr = librosa.load(audio_path, sr=16000)

    # Process input features for the model
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    # Generate prediction
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            task="transcribe",
            use_cache=True,
            max_length=225
        )

    # Decode prediction
    predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    metric_wer = evaluate.load("wer")
#     # Print result immediately
#     print(f"File {i+1}/{len(test_df)}: {audio_path}")
#     print("Predicted:", predicted_text)
#     print("Reference:", reference_text)
#     print("-" * 40)

#     # Store predictions and references for metric calculation
#     predictions.append(predicted_text)
#     references.append(reference_text)


# Normalize predictions and references
pred_str_norm = [twi_financial_normalizer(p) for p in predictions]
label_str_norm = [twi_financial_normalizer(l) for l in references]

# Filter out empty references to avoid errors in WER computation
filtered_predictions = []
filtered_references = []
for p, r in zip(pred_str_norm, label_str_norm):
    if len(r) > 0:  # Only keep pairs where reference is not empty
        filtered_predictions.append(p)
        filtered_references.append(r)

# Compute WER on normalized text
test_wer = 100 * metric_wer.compute(predictions=filtered_predictions, references=filtered_references)

# Compute CER on normalized text
test_cer = 100 * compute_character_error_rate(filtered_predictions, filtered_references)

print("Final Test WER:", test_wer)
print("Final Test CER:", test_cer)

import os
import torch
import librosa
import pandas as pd
import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor


model_checkpoint = "/content/drive/MyDrive/final/checkpoint-1000"
test_csv_path = "/content/fisd-asanti-twi-10p/data.csv"
audio_base_path = "/content/fisd-asanti-twi-10p/audios"

# Load the processor and model from the checkpoint
processor = WhisperProcessor.from_pretrained(model_checkpoint)
model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Load the test CSV into a pandas DataFrame
test_df = pd.read_csv(test_csv_path, delimiter="\t", on_bad_lines='skip')
if "Audio Filepath" not in test_df.columns or "Transcription" not in test_df.columns:
    raise ValueError("The test CSV must have 'Audio Filepath' and 'Transcription' columns.")

# Fix the audio paths
test_df["Audio Filepath"] = test_df["Audio Filepath"].apply(
    lambda x: os.path.join(audio_base_path, os.path.basename(x))
)

# Metrics and Normalizer Setup
metric_wer = evaluate.load("wer")

def twi_financial_normalizer(text: str) -> str:
    # Replace this with your actual normalization logic
    text = text.lower().strip()
    return text

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    import numpy as np
    distance = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        distance[i][0] = i
    for j in range(n+1):
        distance[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            distance[i][j] = min(
                distance[i-1][j] + 1,
                distance[i][j-1] + 1,
                distance[i-1][j-1] + cost
            )
    return distance[m][n]

def compute_character_error_rate(predictions, references):
    total_distance = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        pred = pred.strip()
        ref = ref.strip()
        dist = levenshtein_distance(ref, pred)
        total_distance += dist
        total_chars += len(ref)
    if total_chars == 0:
        return 0.0
    return total_distance / total_chars

# Inference Loop
predictions = []
references = []

for i, row in test_df.iterrows():
    audio_path = row["Audio Filepath"]
    reference_text = row["Transcription"]

    # Load and resample audio
    waveform, sr = librosa.load(audio_path, sr=16000)

    # Process input features for the model
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    # Generate prediction
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            task="transcribe",
            use_cache=True,
            max_length=225
        )

    # Decode prediction
    predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Print result immediately (optional)
    print(f"File {i+1}/{len(test_df)}: {audio_path}")
    print("Predicted:", predicted_text)
    print("Reference:", reference_text)
    print("-" * 40)

    # Store for metric calculation
    predictions.append(predicted_text)
    references.append(reference_text)

##########################################
# Compute WER and CER
##########################################
# Normalize predictions and references
pred_str_norm = [twi_financial_normalizer(p) for p in predictions]
label_str_norm = [twi_financial_normalizer(l) for l in references]

# Filter out empty references to avoid errors in WER computation
filtered_predictions = []
filtered_references = []
for p, r in zip(pred_str_norm, label_str_norm):
    if len(r) > 0:  # Only keep pairs where reference is not empty
        filtered_predictions.append(p)
        filtered_references.append(r)

# Compute WER on normalized text
wer = 100 * metric_wer.compute(predictions=filtered_predictions, references=filtered_references)

# Compute CER on normalized text
cer = 100 * compute_character_error_rate(filtered_predictions, filtered_references)

print("Final Test WER:", wer)
print("Final Test CER:", cer)

import os
import torch
import librosa
import pandas as pd
import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor

##########################################
# User-defined paths (adjust these)
##########################################
model_checkpoint = "/content/drive/MyDrive/final/checkpoint-1000"  # e.g. "/content/drive/MyDrive/whisper-small-tw"
test_csv_path = "/content/data.xlsx"          # e.g. "/content/test_data.csv"
audio_base_path = "/content/fisd-asanti-twi-10p/audios"               # e.g. "/content/test_audios"

##########################################
# Load the processor and model from the checkpoint
##########################################
processor = WhisperProcessor.from_pretrained(model_checkpoint)
model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

##########################################
# Load the test CSV into a pandas DataFrame
##########################################
test_df = pd.read_excel(test_csv_path)
if "Audio Filepath" not in test_df.columns or "Transcription" not in test_df.columns:
    raise ValueError("The test CSV must have 'Audio Filepath' and 'Transcription' columns.")

# Fix the audio paths
test_df["Audio Filepath"] = test_df["Audio Filepath"].apply(
    lambda x: os.path.join(audio_base_path, os.path.basename(x))
)

##########################################
# Metrics and Normalizer Setup
##########################################
metric_wer = evaluate.load("wer")

def twi_financial_normalizer(text: str) -> str:
    # Replace this with your actual normalization logic
    text = text.lower().strip()
    return text

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    import numpy as np
    distance = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        distance[i][0] = i
    for j in range(n+1):
        distance[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            distance[i][j] = min(
                distance[i-1][j] + 1,
                distance[i][j-1] + 1,
                distance[i-1][j-1] + cost
            )
    return distance[m][n]

def compute_character_error_rate(predictions, references):
    total_distance = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        pred = pred.strip()
        ref = ref.strip()
        dist = levenshtein_distance(ref, pred)
        total_distance += dist
        total_chars += len(ref)
    if total_chars == 0:
        return 0.0
    return total_distance / total_chars

##########################################
# Inference Loop
##########################################
predictions = []
references = []

for i, row in test_df.iterrows():
    audio_path = row["Audio Filepath"]
    reference_text = row["Transcription"]

    # Load and resample audio
    waveform, sr = librosa.load(audio_path, sr=16000)

    # Process input features for the model
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    # Generate prediction
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            task="transcribe",
            use_cache=True,
            max_length=225
        )

    # Decode prediction
    predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Print result immediately (optional)
    print(f"File {i+1}/{len(test_df)}: {audio_path}")
    print("Predicted:", predicted_text)
    print("Reference:", reference_text)
    print("-" * 40)

    # Store for metric calculation
    predictions.append(predicted_text)
    references.append(reference_text)

##########################################
# Compute WER and CER
##########################################
# Normalize predictions and references
pred_str_norm = [twi_financial_normalizer(p) for p in predictions]
label_str_norm = [twi_financial_normalizer(l) for l in references]

# Filter out empty references to avoid errors in WER computation
filtered_predictions = []
filtered_references = []
for p, r in zip(pred_str_norm, label_str_norm):
    if len(r) > 0:  # Only keep pairs where reference is not empty
        filtered_predictions.append(p)
        filtered_references.append(r)

# Compute WER on normalized text
wer = 100 * metric_wer.compute(predictions=filtered_predictions, references=filtered_references)

# Compute CER on normalized text
cer = 100 * compute_character_error_rate(filtered_predictions, filtered_references)

print("Final Test WER:", wer)
print("Final Test CER:", cer)