import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to analyze data and generate plots
def analyze_data(file, user_prompt):
    if file is None:
        return "Please upload a CSV file.", None

    # Load CSV into DataFrame
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"Error reading CSV file: {e}", None

    # Basic data info
    info = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
    info += f"Columns: {list(df.columns)}\n\n"

    # Generate plot (correlation heatmap as example)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plot_path = "correlation_heatmap.png"
    plt.savefig(plot_path)
    plt.close()

    # Call LLM for user query
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful data analysis assistant. Answer based on the dataset."},
                {"role": "user", "content": f"Data columns: {list(df.columns)}. {user_prompt}"}
            ]
        )
        llm_answer = response.choices[0].message.content
    except Exception as e:
        llm_answer = f"Error calling OpenAI API: {e}"

    return info + "\n\n" + llm_answer, plot_path


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📊 Data Analyzer with GPT-4o")
    gr.Markdown("Upload a CSV file, ask questions, and get insights + plots.")

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        user_input = gr.Textbox(label="Ask a question about the data")

    with gr.Row():
        output_text = gr.Textbox(label="Analysis & Answer", lines=10)
        output_plot = gr.Image(label="Generated Plot")

    submit_btn = gr.Button("Analyze")

    submit_btn.click(
        fn=analyze_data,
        inputs=[file_input, user_input],
        outputs=[output_text, output_plot]
    )

if __name__ == "__main__":
    demo.launch()
