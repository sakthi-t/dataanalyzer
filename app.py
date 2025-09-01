import os
import uuid
import shutil
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# -------------------------------
# Session memory
# -------------------------------
chat_history = {}

def ask_question(session_id, question, df):
    """Q&A with GPT-4o over the uploaded dataset."""
    if session_id not in chat_history:
        chat_history[session_id] = []
    
    # Summarize dataset schema (only first rows to avoid token overflow)
    schema_info = f"Columns: {list(df.columns)}\n\nFirst rows:\n{df.head().to_string()}"

    # Build conversation
    messages = [
        {"role": "system", "content": "You are a helpful data analysis assistant. Answer based only on the dataset provided."},
        {"role": "user", "content": f"Dataset info:\n{schema_info}"}
    ]
    for msg in chat_history[session_id]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    # GPT call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )

    # ✅ Extract correctly from new SDK
    answer = response.choices[0].message.content

    # Update session history
    chat_history[session_id].append({"role": "user", "content": question})
    chat_history[session_id].append({"role": "assistant", "content": answer})

    return answer


def generate_plot(session_id, plot_request, df):
    """Generate a matplotlib plot based on user request."""
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(tempfile.gettempdir(), filename)

    try:
        req = plot_request.lower()

        if "heatmap" in req:
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            corr = df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        
        elif "hist" in req:
            df.hist(figsize=(8, 6))
        
        elif "scatter" in req:
            cols = df.select_dtypes(include="number").columns
            if len(cols) >= 2:
                plt.figure(figsize=(6, 4))
                plt.scatter(df[cols[0]], df[cols[1]])
                plt.xlabel(cols[0])
                plt.ylabel(cols[1])
            else:
                return None
        
        elif "bar" in req:
            df.select_dtypes(include="number").mean().plot.bar(figsize=(8, 6))
        
        else:
            df.plot(figsize=(8, 6))
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        return filepath

    except Exception as e:
        return f"Error generating plot: {str(e)}"


def download_zip(session_id):
    """Bundle all plots and answers into a zip file and delete temp files."""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tempfile.gettempdir(), f"results_{session_id}.zip")

    # Save chat history
    history_file = os.path.join(temp_dir, "chat_history.txt")
    with open(history_file, "w", encoding="utf-8") as f:
        for msg in chat_history.get(session_id, []):
            f.write(f"{msg['role']}: {msg['content']}\n")

    # Copy plots
    for fname in os.listdir(tempfile.gettempdir()):
        if fname.endswith(".png"):
            shutil.copy(os.path.join(tempfile.gettempdir(), fname), temp_dir)

    # Create zip
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', temp_dir)

    # Cleanup
    shutil.rmtree(temp_dir)

    return zip_path


# -------------------------------
# Gradio Interface
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 📊 AI Data Analyzer (GPT-4o + Gradio)")

    session_id = gr.State(str(uuid.uuid4()))
    df_state = gr.State()

    with gr.Row():
        csv_upload = gr.File(label="Upload CSV", file_types=[".csv"])
        download_btn = gr.Button("Download ZIP")
        zip_out = gr.File()

    with gr.Tab("Q&A"):
        question = gr.Textbox(label="Ask a question about the dataset")
        answer = gr.Textbox(label="Answer")
        ask_btn = gr.Button("Submit Question")

    with gr.Tab("Charts"):
        plot_request = gr.Textbox(label="Request a chart (e.g., histogram, scatter)")
        plot_output = gr.Image(label="Generated Plot")
        plot_btn = gr.Button("Generate Plot")

    # Upload CSV
    def load_csv(file):
        df = pd.read_csv(file.name)
        return df, f"CSV Loaded. Shape: {df.shape}"

    status = gr.Textbox(label="Status")
    csv_upload.change(load_csv, inputs=[csv_upload], outputs=[df_state, status])

    # Q&A
    ask_btn.click(ask_question, inputs=[session_id, question, df_state], outputs=[answer])

    # Plot generation
    plot_btn.click(generate_plot, inputs=[session_id, plot_request, df_state], outputs=[plot_output])

    # Download ZIP
    download_btn.click(download_zip, inputs=[session_id], outputs=[zip_out])

if __name__ == "__main__":
    demo.launch()
