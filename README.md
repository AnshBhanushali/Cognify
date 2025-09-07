# Cognify

**AI-Powered Human-in-the-Loop Labeling System**

Cognify is a **multimodal data labeling platform** that fuses **machine learning models, embeddings, and human validation** to streamline dataset creation.

It provides **smart label suggestions**, **live retraining**, and **dataset visualization**, enabling a feedback loop where AI and humans co-evolve better labels together.

---

## 🌟 Core Highlights

* 🔹 **Multimodal Support** – Upload **images** or record **audio**, both processed into embeddings.
* 🔹 **AI Label Suggestions** from:

  * **CLIP** – Vision-language embeddings
  * **ChromaDB** – Vector similarity search
  * **GPT-4V** (planned/experimental) – High-level semantic reasoning
* 🔹 **On-the-fly Classifier** – Incremental **k-NN (scikit-learn)** retrains as you save labels.
* 🔹 **Audio Intelligence** – Faster-Whisper transcription, confidence scoring, broken word detection, diarization, and denoising.
* 🔹 **Dataset Management** – Save to ChromaDB, explore entries, download JSON/CSV.
* 🔹 **Visualization** – Embeddings projected in 2D with UMAP/t-SNE (planned).
* 🔹 **Human-in-the-Loop Workflow** – Confidence-based active learning: uncertain samples surfaced first.

---

## 🛠️ Tech Stack

### ⚡ Frontend (cognify-frontend)

* **React (Vite + TypeScript)**
* **TailwindCSS** + **shadcn/ui** (modern, clean UI)
* **React Router v6** (multi-page flow)
* **Recharts** (embedding visualization)

### ⚡ Backend (cognify-backend)

* **FastAPI** (Python) – REST API
* **Uvicorn** – ASGI server
* **CORS** – configured for Vercel ↔ Render

### ⚡ ML / AI Models & Tools

* **CLIP** (OpenAI) – Image embeddings
* **ChromaDB** – Vector database for retrieval
* **faster-whisper** – Speech-to-text + word confidence
* **scikit-learn** – k-Nearest Neighbors (incremental classifier)
* **GPT-4V** – (planned) semantic refinement of labels
* **Librosa + Noisereduce** – Audio preprocessing & denoising
* **UMAP / t-SNE** – (planned) embedding projection

---

## 🧩 System Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e23f493f-2720-4355-9cfe-e857db75c67a" />


---

## ⚙️ Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AnshBhanushali/Cognify.git
cd Cognify
```

### 2️⃣ Backend Setup (FastAPI)

```bash
cd cognify-backend
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs on: `http://localhost:8000`

### 3️⃣ Frontend Setup (React)

```bash
cd cognify-frontend
npm install
npm run dev
```

Frontend runs on: `http://localhost:5173`

---

## 🔑 API Endpoints

| Endpoint                 | Method | Description                                         |
| ------------------------ | ------ | --------------------------------------------------- |
| `/upload/image`          | POST   | Upload image → embedding + suggestions              |
| `/upload/audio`          | POST   | Upload audio → transcript + embedding + suggestions |
| `/confirm`               | POST   | Save labeled image                                  |
| `/confirm_audio`         | POST   | Save labeled audio                                  |
| `/predict`               | POST   | Run k-NN classifier on embedding                    |
| `/retrain`               | POST   | Retrain classifier on dataset                       |
| `/dataset`               | GET    | List dataset entries                                |
| `/dataset/download/json` | GET    | Download dataset as JSON                            |
| `/dataset/download/csv`  | GET    | Download dataset as CSV                             |
| `/embeddings/all`        | GET    | Export embeddings for visualization                 |

---

## 📊 Dataset Page (Frontend)

* **Refresh Dataset** → pull latest entries
* **Retrain Model** → hit `/retrain` endpoint
* **Download** → JSON / CSV export
* **Visualization** → Scatterplot of embeddings (Recharts)

---

## 🚀 Roadmap

### ✅ Sprint 1 – Core Foundations

* Upload (Image/Audio)
* CLIP embeddings + Whisper transcription
* Dataset save/download

### ✅ Sprint 2 – Feedback & Training

* k-NN classifier (scikit-learn)
* Retraining button
* Embedding visualization

### 🔜 Sprint 3 – Enhancements

* Active Learning loop (uncertain samples)
* Bulk/Semi-Automatic labeling
* Hierarchical Labels (Car → Honda → Civic)
* Confidence Heatmap (UMAP/t-SNE)
* Region-based labeling (SAM, Segment Anything)
* Text-Prompt Search ("find all images like a dog")

---

## 🌐 Deployment

* **Backend** → Render (FastAPI)
* **Frontend** → Vercel (React + Vite)
* **CORS** → Configured to allow cross-origin API calls

---

## 👨‍💻 Contributors

* **Ansh Bhanushali** – Full-stack developer, ML engineer, and project lead

---

## 📜 License

MIT License – free to use, modify, and distribute.
