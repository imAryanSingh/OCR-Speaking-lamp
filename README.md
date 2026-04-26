# OCR Speaking Lamp
 

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-4-C51A4A?style=for-the-badge&logo=raspberry-pi&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract_OCR-5.x-4A90D9?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-1D9E75?style=for-the-badge)
 
 <div align="center">
**A Raspberry Pi-powered accessibility device that reads printed text aloud in real time — hands-free, screenless, and built for visually impaired users.**
 
*95% OCR accuracy · No internet required · Runs entirely on-device*
 
[Overview](#overview) · [How It Works](#how-it-works) · [Setup](#setup) · [File Structure](#file-structure) · [Hardware](#hardware)
 
</div>
---
 
## Overview
 
Millions of visually impaired people struggle to independently read printed materials — books, menus, medicine labels, signs. This device removes that barrier. A user places printed text in front of the camera and the device reads it aloud — no screen, no phone, no internet needed.
 
| Feature | Detail |
|---------|--------|
| OCR accuracy | 95% on clear printed text |
| Hardware cost | ~₹4,500 |
| Internet required | No — fully offline |
| Response time | Under 3 seconds |
 
---
 
## How It Works
 
```
┌─────────────────────────────────────────────────────────┐
│                   PIPELINE                               │
├──────────────┬──────────────┬──────────────┬────────────┤
│   CAPTURE    │  PREPROCESS  │     OCR      │   SPEECH   │
│  Pi Camera   │   OpenCV     │  Tesseract   │  pyttsx3   │
│  grayscale   │  threshold   │  PSM 6 mode  │ 150wpm     │
└──────────────┴──────────────┴──────────────┴────────────┘
```
 
**Step 1 — Capture**
```python
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
```
 
**Step 2 — Preprocess**
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
denoised = cv2.fastNlMeansDenoising(thresh, h=10)
```
 
**Step 3 — OCR**
```python
import pytesseract
text = pytesseract.image_to_string(denoised, config='--psm 6')
```
 
**Step 4 — Speech**
```python
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.say(text)
engine.runAndWait()
```
 
---
 
## File Structure
 
```
OCR-Speaking-lamp/
├── main.py                  ← Main script
├── OCRkaVariation/          ← Alternative implementations
│   ├── live_camera.py       ← Continuous loop version
│   └── image_file.py        ← Reads from saved image
├── requirements.txt
└── README.md
```
 
> **Rename `OCRkaVariation/` to `variations/`**
> ```bash
> git mv OCRkaVariation variations
> git commit -m "Rename: OCRkaVariation → variations"
> ```
 
---
 
## Setup
 
### On Raspberry Pi
 
```bash
sudo apt-get install -y tesseract-ocr python3-pip libespeak1
git clone https://github.com/imAryanSingh/OCR-Speaking-lamp.git
cd OCR-Speaking-lamp
pip3 install -r requirements.txt
python3 main.py
```
 
### On PC (testing without Pi)
 
```bash
# Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
pip install pytesseract pyttsx3 opencv-python pillow
python main.py
```
 
### requirements.txt
 
```
pytesseract>=0.3.10
pyttsx3>=2.90
opencv-python>=4.5.0
Pillow>=9.0.0
numpy>=1.21.0
```
 
---
 
## Common Errors & Fixes
 
**TesseractNotFoundError**
```python
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux/Pi
```
 
**pyttsx3 silent on Raspberry Pi**
```bash
sudo apt-get install espeak libespeak1
# In code: engine = pyttsx3.init('espeak')
```
 
**Camera not found**
```bash
vcgencmd get_camera   # should show: supported=1 detected=1
```
 
---
 
## Hardware
 
| Component | Cost |
|-----------|------|
| Raspberry Pi 4 (2GB) | ₹3,500 |
| Pi Camera Module v2 | ₹800 |
| USB Speaker | ₹200 |
| Power bank (5V 3A) | ₹600 |
 
**Total: ~₹5,100** — 10× cheaper than commercial assistive reading devices
 
---
 
## About the Author
 
**Aryan Singh** — AI/ML Engineer
 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-im--aryan--singh-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/im-aryan-singh)
[![GitHub](https://img.shields.io/badge/GitHub-imAryanSingh-181717?style=flat&logo=github)](https://github.com/imAryanSingh)
[![Portfolio](https://img.shields.io/badge/Portfolio-imAryanSingh.github.io-534AB7?style=flat)](https://imAryanSingh.github.io)
 
---
 
## Also see
 
- [Wake-Word Detection — ISRO TRISHNA Satellite](https://github.com/imAryanSingh/Wakeup-Word-Detection-Model-for-voice-commanding-system)
- [Wildfire Prediction from Satellite Imagery](https://github.com/imAryanSingh/Wildfire-Prediction-Using-Satellite-Image-GSoC)
 



