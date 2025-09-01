# AI Data Analyzer (Q&A + Visualization)

This project is an **AI-powered CSV Data Analyzer** built with **Gradio** and **OpenAI GPT-4o**.  
It allows users to upload CSV files, ask natural language questions about the data, and generate **visualizations** (charts/plots).  

## 🚀 Features
- Upload any CSV file for analysis.  
- Ask **follow-up questions** in natural language with **chat memory**.  
- Generate **plots/visualizations** (Matplotlib). Limited to bar chart, histogram, scatter plot, defaults to simple plot.   
- Download all generated outputs (charts + answers) as a **ZIP file**.  
- Auto-cleans up server memory after download for efficiency.  
- Secure API key management using **dotenv**.  
- Fully compatible with **Hugging Face Spaces** (runs on Gradio, not Streamlit).  

## 🛠️ Tech Stack
- **Python 3.10+**  
- **Gradio** – for building the web interface  
- **Pandas** – data handling  
- **Matplotlib** – visualization  
- **OpenAI GPT-4o** – LLM for Q&A over CSV data  
- **dotenv** – environment variable management  



