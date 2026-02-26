# 🧾 Receipt Information Extraction API (Flask + Deep Learning)

This project provides a Flask web application that performs end-to-end receipt information extraction from images.

The system combines:

Text region detection (object detection)

Text recognition (OCR)

Information extraction (NER / field classification)

It returns structured fields such as:

company

date

address

total

# Features

✅ Upload receipt images via web or API
✅ Automatic text detection and OCR
✅ Structured information extraction
✅ Lazy model loading (load once, reuse)
✅ Fallback regex for date and total
✅ Debug metadata output
✅ GPU supported (optional)

# Pipeline Overview

Input Image
     ↓
[Detection Model]
  Faster R-CNN
  → find text regions
     ↓
[Recognition Model]
  TrOCR
  → convert image crops to text
     ↓
[Information Extraction Model]
  LayoutLMv3
  → classify words into fields
     ↓
Structured JSON Output


# Models Used

| Task                   | Model                                |
| ---------------------- | ------------------------------------ |
| Detection              | Faster R-CNN ResNet50 FPN            |
| Recognition            | TrOCR (microsoft/trocr-base-printed) |
| Information Extraction | LayoutLMv3 (token classification)    |


## Task 1 – Text Detection

We evaluate the detection model using Precision (P), Recall (R), and F1-score at two IoU thresholds (0.5 and 0.7).

Validation Set

| IoU | Precision | Recall | F1     |
| --- | --------- | ------ | ------ |
| 0.5 | 0.9501    | 0.9623 | 0.9561 |
| 0.7 | 0.8974    | 0.9090 | 0.9032 |

Test Set
| IoU | Precision | Recall | F1     |
| --- | --------- | ------ | ------ |
| 0.5 | 0.9488    | 0.9641 | 0.9564 |
| 0.7 | 0.9012    | 0.9158 | 0.9085 |

Analysis

At IoU@0.5, the detector achieves F1 ≈ 0.956 on both VAL and TEST.

At stricter IoU@0.7, performance slightly decreases but remains strong (F1 ≈ 0.90).

The small gap between VAL and TEST indicates good generalization.

Recall is consistently slightly higher than precision, meaning:

The model prefers detecting more boxes.

It rarely misses text regions.

Performance stability across datasets suggests the detector is robust and well-trained.

Overall, the detection component is strong and unlikely to be the bottleneck in the pipeline.

## Task 2 – Text Recognition (OCR)

Recognition performance is measured using:

CER (Character Error Rate)

WER (Word Error Rate)

Exact Accuracy

Validation Set

CER = 0.02095
WER = 0.08947
Exact Accuracy = 0.8525

Test Set

CER = 0.02201
WER = 0.09255
Exact Accuracy = 0.86188

Analysis

CER ≈ 2.2% on TEST → extremely low character-level error.

WER ≈ 9.3%, meaning most words are recognized correctly.

Exact match accuracy ≈ 86%, which is strong for receipt OCR.

Key observations:

Very small gap between VAL and TEST → no overfitting.

Low CER shows the TrOCR fine-tuning worked effectively.

Errors are likely caused by:

Very small fonts

Blurry regions

Special characters or receipt noise

The OCR stage is highly accurate and reliable.

## Task 3 – Information Extraction (SROIE-style Field Evaluation)

Evaluation is done using field-level exact match micro metrics on the TEST set.

TP = 1156
FP = 120
FN = 232
Precision = 0.9060
Recall = 0.8329
F1 = 0.8679
Analysis

Precision ≈ 90.6%

Recall ≈ 83.3%

F1 ≈ 86.8%

This means:

When the model predicts a field → it is usually correct (high precision).

Some fields are still missed (lower recall).

The main source of FN likely comes from:

Missing OCR words

Incorrect token alignment

LayoutLM classification boundary errors

Compared to OCR, the IE stage is currently the most challenging part of the pipeline.


# Project Structure
project/
│
├── app.py
├── uploads/
├── templates/
│   └── index.html
│
├── best_det.pt
├── best_rec.pt
└── layoutlmv3_sroie_out/

# How to run 

pip install -r requirements.txt

python app.py 

http://localhost:5000

using python 3.12.12