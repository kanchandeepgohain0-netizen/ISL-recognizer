# ISL-recognizer

An Indian Sign Language (ISL) recognition system incorporating real-time video capture, keypoint extraction, LSTM model inference, and FAISS-based similarity search.

## Project Structure

```text
ISL-recognizer/
│
├── backend/                           # 🌐 SERVER: Handles WebSockets & Inference
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application entry point
│   │   ├── websocket_handler.py       # Logic for 15s ping-pong and connection state
│   │   ├── inference.py               # Loads the LSTM model and runs FAISS similarity search
│   │   └── config.py                  # Server configuration (confidence thresholds, ports)
│   ├── models_db/                     # Stored PyTorch weights and the FAISS index database
│   └── requirements.txt               # Server dependencies (fastapi, uvicorn, faiss, etc.)
│
├── ml_pipeline/                       # 🧠 MACHINE LEARNING: Training and Data Preparation
│   ├── data/
│   │   ├── raw_videos/                # Original video files (~700 samples)
│   │   └── processed_keypoints/       # Saved .npy arrays after MediaPipe + Normalization
│   ├── notebooks/                     # Jupyter notebooks for EDA and testing pipeline steps
│   ├── src/
│   │   ├── extract_keypoints.py       # MediaPipe script (filters face/legs, preserves 3D hand coordinates)
│   │   ├── normalize.py               # Calculates shoulder midpoint and normalizes coordinates
│   │   ├── augment.py                 # Applies Gaussian noise and temporal stretching
│   │   ├── dataset.py                 # Data loader for the training loop
│   │   ├── model.py                   # LSTM neural network architecture definition
│   │   ├── train.py                   # Script to train the LSTM with Triplet Loss
│   │   └── build_faiss_index.py       # Script to embed reference samples and save index DB
│   └── requirements.txt               # ML dependencies (mediapipe, torch, numpy, etc.)
│
├── frontend/                          # 📱 FRONTEND CLIENT: React + Vite application
│   ├── src/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
└── README.md                          # Project documentation and setup instructions
```

## Setup & Running

- **Backend**: Navigate to `backend/`, install packages from `requirements.txt`, and run `main.py` to start the FastAPI server.
- **Frontend**: Navigate to `frontend/`, run `npm install` and `npm run dev` to start the development server.
- **ML Pipeline**: Setup packages from `ml_pipeline/requirements.txt` to run keypoint extraction and training scripts in `ml_pipeline/src/`.