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

## 🧠 How It Works
- Select an LLM provider (local or cloud)
- Upload your dataset and select a target column
- The app:
  - Auto-summarizes the data
  - Performs exploratory data analysis
  - Sends data summary to an LLM to receive smart suggestions
  - Falls back to rule-based logic when no LLM is used

## 📂 Folder Structure
```
/app.py
/requirements.txt
/README.md
/config.json (optional - for storing API keys locally)
```

## 🧰 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Run the App
```bash
streamlit run app.py
```

## 💡 Credits
Developed by [Your Name]  
Version: v1.0.0 – First public beta release.

## 🔗 Connect
- LinkedIn: [your-profile-link]
- GitHub: [your-repo-link]
