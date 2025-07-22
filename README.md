# 📊 Analyzer AI Dashboard (v1.0.0)

An interactive Streamlit app that uses AI to analyze datasets, uncover insights, and support data-driven decision-making. Supports both local and cloud LLMs (OpenAI, Groq, Ollama).

## 🚀 Features
- Upload your own CSV dataset
- Select target (label) column and desired outcome
- Automatic summary, statistics, and visualizations
- Feature correlation analysis with heatmaps
- **LLM Analysis** (Choose: OpenAI, Groq, or Local/Ollama)
- **Rule-based fallback** if no LLM is used
- Secure API key entry (OpenAI & Groq)
- Test connection to local LLMs
- Works both **online and offline**

## 📦 Run the App

### ▶️ Option 1: Run on Streamlit Cloud (Online)
Use OpenAI or Groq for cloud-based LLM insights.

🌐 [Launch the App](https://analyzer-ai-dashboard-kqqpqjmemqrnuvrcqeva9s.streamlit.app)

> *Local LLM is not available on the cloud version.*

---

### 💻 Option 2: Run Locally with Ollama (Offline)
Use this if you want full control and local LLM support.

#### 1. Clone or download the release:
```bash
git clone https://github.com/Dhafer9855/analyzer-ai-dashboard
cd analyzer-ai-dashboard
```

Or [Download the ZIP Release](https://github.com/Dhafer9855/analyzer-ai-dashboard/releases)

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run the app
```bash
streamlit run app.py
```

#### 4. (Optional) Enable Local LLM
To use a local model with Ollama:
```bash
ollama serve
ollama pull llama3
```

Then in the app:
- Choose provider: `local`
- ✅ Check: Use Local LLM
- Enter model name: `llama3`

---

## 📂 Folder Structure
```
/app.py
/requirements.txt
/README.md
/config.json (optional - for storing API keys locally)
```

## 💡 Credits
Developed by Dhafer9855  
Version: v1.0.0 – First public beta release.

## 🔗 Connect
- LinkedIn: [your-profile-link]
- GitHub: [https://github.com/Dhafer9855/analyzer-ai-dashboard](https://github.com/Dhafer9855/analyzer-ai-dashboard)
