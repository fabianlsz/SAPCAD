---


# SAPCAD (Smart Architectural Planning with CAD Automation)

SAPCAD is a multimodal AI prototype designed to integrate natural language processing with Building Information Modeling (BIM). The system ingests architectural IFC files alongside regulatory documents (PDF norms), enabling automated interpretation of both building geometry and compliance requirements. Through a conversational interface, users can submit requests that are mapped into CAD-level modifications. These updates are applied directly to the underlying model and visualized in a side-by-side 3D viewer. By combining LLM-driven reasoning, structured CAD parsing, and regulation-aware design rules, SAPCAD demonstrates how AI can accelerate architectural workflows, improve regulatory compliance, and reduce the overhead of manual model editing.

https://github.com/user-attachments/assets/c4c5e1c2-1bdc-411f-a5a3-9705f511a58e



---

---

# SAPCAD – Backend & Frontend Setup (with Ollama + Llama 3.2)

This guide explains how to set up **SAPCAD** locally with a **FastAPI backend**, a **React frontend**, and **Ollama** for LLM integration.

---

## Prerequisites

Make sure the following are installed on your system:

* [Python 3.9+](https://www.anaconda.com/products/distribution) (recommended via Anaconda)
* [Node.js 18+](https://nodejs.org/en)
* [Ollama](https://ollama.com/download) (for local LLM inference)

  * Quick intro: [YouTube – Ollama Setup](https://www.youtube.com/watch?v=92_yb31Bqzk&ab_channel=AleksandarHaberPhD)

---

## Backend Setup (FastAPI + Ollama)

Here’s your backend setup guide for FastAPI + Ollama, updated to include the initial Conda environment creation step, while preserving your original structure and formatting:

***

## Backend Setup (FastAPI + Ollama)

1. **Create your Conda environment**  
   *(Run this step only if the environment doesn’t already exist)*
   ```bash
   conda create -n sapcad python=3.11
   ```

2. **Activate your Conda environment**
   ```bash
   conda activate sapcad
   ```

3. **Install backend dependencies**  
   From the backend project folder:
   ```bash
   pip install fastapi uvicorn httpx
   ```

4. **Start Ollama with the Llama 3.2 model**
   * Check installed models:
     ```bash
     ollama list
     ```
   * If `llama3.2` is available, run:
     ```bash
     ollama run llama3.2
     ```
   Keep this terminal open while working — Ollama must stay running.

5. **Run the FastAPI backend**  
   From the folder containing `main.py`:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will be available at: [http://localhost:8000](http://localhost:8000) .

---

## Frontend Setup (React)

1. **Install dependencies**

   From the frontend project folder:

   ```bash
   npm install
   ```

2. **Start the development server**

   ```bash
   npm run dev
   ```

   The frontend will be available at: [http://localhost:5173](http://localhost:5173)

---

## Troubleshooting

| Issue                        | Solution                                                             |
| ---------------------------- | -------------------------------------------------------------------- |
| `'ollama' is not recognized` | Make sure Ollama is installed and added to your system PATH.         |
| CORS errors                  | Update FastAPI CORS settings to allow `http://localhost:5173`.       |
| Model not found              | Run `ollama list` and confirm the model name is `llama3.2`.          |
| Backend fetch fails          | Ensure Ollama is running and accessible at `http://localhost:11434`. |

---

## Notes

* Start **Ollama** before running the backend.
* If frontend or backend ports are changed, update **CORS settings** and API URLs.
* This setup is intended for **local development**. For production, further steps are needed (reverse proxy, HTTPS, security hardening, etc.).

---
