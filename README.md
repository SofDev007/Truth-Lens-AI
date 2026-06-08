# 🔍 TruthLens AI

### Real-Time Multi-Modal Misinformation & Deepfake Detection Platform

TruthLens AI is an AI-powered platform designed to detect misinformation, manipulated content, and deepfakes across multiple media formats. Developed during a hackathon, the system combines Explainable AI (XAI), Natural Language Processing (NLP), and Computer Vision techniques to provide transparent and trustworthy credibility assessments for text, images, videos, and web content.

---

## 🚀 Overview

In today's digital world, fake news and deepfakes spread faster than ever. TruthLens AI helps users verify content authenticity by analyzing multiple input sources and providing instant, explainable verdicts backed by AI-driven reasoning.

The platform supports:

✅ News Articles
✅ Social Media Posts
✅ WhatsApp Forwards
✅ Images
✅ Videos
✅ Website URLs

---

## ✨ Key Features

### 📝 Text Misinformation Detection

Analyze articles, posts, and forwarded messages using NLP-based credibility assessment.

**Capabilities**

* Linguistic pattern analysis
* Suspicious phrase detection
* Claim verification workflow
* Hindi & Hinglish support
* Confidence scoring

---

### 🖼️ Image Deepfake Detection

Identify AI-generated and manipulated images using forensic analysis techniques.

**Detection Methods**

* GAN fingerprint analysis
* Noise inconsistency detection
* Edge and boundary artifact inspection
* Frequency-domain analysis

---

### 🎥 Video Deepfake Detection

Perform frame-level analysis to identify manipulated video content.

**Capabilities**

* Temporal consistency analysis
* Facial artifact detection
* Frame anomaly identification
* Deepfake probability scoring

---

### 🌐 URL Credibility Checker

Evaluate websites and news articles instantly.

**Features**

* Domain reputation verification
* Source credibility analysis
* Content extraction and evaluation
* Unverified source detection

---

### 🧠 Explainable AI (XAI)

Unlike traditional black-box systems, TruthLens AI explains every decision.

Each verdict includes:

* Credibility score
* Confidence percentage
* Risk indicators
* Human-readable reasoning

---

### 📊 Real-Time Admin Dashboard

Monitor platform activity and misinformation trends through a live analytics dashboard.

**Dashboard Insights**

* Global platform statistics
* Trending misinformation keywords
* Recent submissions
* Detection summaries
* Usage analytics

---

## 🛠️ Technology Stack

### Backend

* FastAPI (Python)
* SQLite Database

### Frontend

* HTML5
* CSS3 (Glassmorphism UI)
* Vanilla JavaScript

### AI & Processing

* OpenCV
* Pillow (PIL)
* NumPy
* SciPy
* NLP APIs (Gemini / Groq Integration)

---

## 📂 Project Architecture

```text
TruthLens-AI/
│
├── backend/
│   ├── api_routes/
│   ├── database/
│   ├── models/
│   ├── utils/
│   ├── main.py
│   └── requirements.txt
│
└── frontend/
    ├── app.js
    ├── index.html
    ├── dashboard.html
    └── style.css
```

## ⚡ Quick Setup

### 1️⃣ Backend Setup

```bash
cd backend

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Configure API credentials inside the `.env` file if external services are used.

---

### 2️⃣ Frontend Setup

```bash
cd frontend

python -m http.server 3000
```

Access the application:

```text
Analyzer UI:
http://localhost:3000

Admin Dashboard:
http://localhost:3000/dashboard.html
```

---

## 🎯 Use Cases

* Fake News Detection
* Deepfake Identification
* Social Media Content Verification
* Media Authenticity Analysis
* Educational Demonstrations
* Cyber Awareness Campaigns
* Journalism & Fact-Checking Support

---

## 🔮 Future Enhancements

* Real-time browser extension
* Social media integration
* Advanced deep learning models
* Multi-language support
* Cloud deployment
* Community fact-checking network

---

## ⚠️ Disclaimer

TruthLens AI is currently a prototype developed for hackathon demonstration purposes. Detection results are generated using AI-assisted heuristics and analytical models and should not be considered a substitute for professional digital forensic investigations.

---

### 👨‍💻 Developed by

**Arnav Jaiswal**
B.Tech Information Technology
Medi-Caps University

*"Empowering users to verify information before believing it."*
