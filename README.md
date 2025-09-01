# SAPCAD Backend + Frontend Setup (Ollama Llama 3.2 Integration)

This project connects a React frontend and a FastAPI backend with a locally running Llama 3.2 model via Ollama.

---

## 📦 Prerequisites

Make sure you have the following installed:

- **Python 3.9+** (preferably managed with [Anaconda](https://www.anaconda.com/products/distribution))
- **Node.js 18+**
- **Ollama** (for local LLM inference) → [https://ollama.com/download](https://ollama.com/download)

---

## ⚙️ Backend Setup (FastAPI + Ollama)

### 1. Activate the Conda Environment

```bash
conda activate sapcad
```

> If the environment does not exist, please create it first or ask the team.

### 2. Install Python Dependencies

Inside the backend project directory:

```bash
pip install fastapi uvicorn httpx
```

> *(Only needed once unless dependencies are updated.)*

### 3. Start Ollama and Llama 3.2 Model

Check installed models:

```bash
ollama list
```

If Llama 3.2 is available, run:

```bash
ollama run llama3.2
```

Keep this terminal session running while developing.

### 4. Start the Backend Server

In the backend project folder where `main.py` exists:

```bash
uvicorn main:app --reload
```

The backend will be available at:

```
http://localhost:8000
```

---

## 🎨 Frontend Setup (React)

### 1. Install Frontend Dependencies

Navigate to the frontend project directory:

```bash
npm install
```

### 2. Run Frontend Development Server

```bash
npm run dev
```

The frontend will be available at:

```
http://localhost:5173
```

---

## 🔗 How the System Works

1. User submits a prompt via the frontend.
2. The frontend sends a `POST` request to:
   ```
   http://localhost:8000/llm
   ```
3. The backend forwards the prompt to the Llama 3.2 model via Ollama.
4. The model generates a response, which the backend returns to the frontend for display.

---

## 🛠 Troubleshooting

| Issue | Solution |
|:-----|:---------|
| `'ollama' is not recognized` | Ensure Ollama is installed and added to system PATH. |
| CORS errors | Confirm CORS settings in the backend allow `http://localhost:5173`. |
| Model not found | Check `ollama list` and use the correct model name (`llama3.2`). |
| Backend fetch fails | Ensure Ollama is running and accessible at `http://localhost:11434`. |

---

## ✅ Summary Commands

```bash
# Backend
conda activate sapcad
pip install fastapi uvicorn httpx  # if not already installed
ollama run llama3.2
uvicorn main:app --reload

# Frontend
npm install
npm run dev
```

---

## ✍️ Notes

- Always ensure Ollama is running before starting the backend.
- If frontend or backend ports change, update CORS settings and API URLs accordingly.
- This setup is intended for **local development**. Production deployment will require further adjustments (reverse proxy, HTTPS, security hardening, etc.).

---

# 🚀 Happy Developing!

# LLaMA 3.2 LoRA Fine-tuning Script

## Requirements
- Python 3.8+
- GPU with at least 16GB VRAM (recommended for large models)
- Internet connection for downloading models from HuggingFace (unless using local models)

### Python Dependencies
Install all required packages with:
```bash
pip install -r requirements.txt
```

## Dataset Format
- Training and validation datasets must be CSV files with a column named `text`.

## Usage
### Training
```bash
python llama_3_2_pretrain.py --mode train --dataset_path ./datasets/my_dataset/data.csv --output_dir ./models/llama3_lora
```

### Training with Validation and Resume
```bash
python llama_3_2_pretrain.py --mode train --dataset_path train.csv --val_dataset_path val.csv --resume_from_checkpoint ./models/llama3_lora/checkpoint-1000
```

### Inference
```bash
python llama_3_2_pretrain.py --mode infer --output_dir ./models/llama3_lora
```

## Notes
- If you get out-of-memory errors, reduce the batch size or use a smaller model.
- For first-time model use, ensure you have an internet connection to download weights from HuggingFace.
- CUDA and GPU drivers must be installed for GPU acceleration.

## Distributed/Multi-GPU Training

The script supports distributed and multi-GPU training out of the box via HuggingFace Trainer.

To train on multiple GPUs using PyTorch:
```bash
torchrun --nproc_per_node=4 llama_3_2_pretrain.py --mode train --dataset_path ./datasets/my_dataset/data.csv --output_dir ./models/llama3_lora
```

Or using HuggingFace Accelerate:
```bash
accelerate launch llama_3_2_pretrain.py --mode train --dataset_path ./datasets/my_dataset/data.csv --output_dir ./models/llama3_lora
```

Adjust `--nproc_per_node` to the number of GPUs available on your machine.
