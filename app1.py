"""
AI CSV Analyst (single-file Gradio app)

Features:
- Upload a CSV once per session
- Ask multiple natural-language queries (LLM interprets)
- Generate multiple plots (LLM suggests plot parameters; backend validates & draws)
- Keep session memory (df, queries, answers, plots)
- Download all results as a ZIP (CSV + queries/answers + plots)
- Auto-delete the ZIP a short while after it's created (to avoid storage buildup)
"""

import os
import time
import json
import uuid
import shutil
import zipfile
import tempfile
import threading
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import openai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# --- CONFIG ---
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"  # as requested
ZIP_DELETE_DELAY = 30  # seconds after which the zip file will be removed

# --- Helpers for scheduling file removal ---
def _delayed_remove(path, delay):
    try:
        time.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def schedule_delete(path, delay=ZIP_DELETE_DELAY):
    t = threading.Thread(target=_delayed_remove, args=(path, delay), daemon=True)
    t.start()

# --- Session state initializer ---
def init_state():
    sid = uuid.uuid4().hex[:8]
    session_dir = os.path.join("sessions", sid)
    os.makedirs(session_dir, exist_ok=True)
    return {
        "id": sid,
        "dir": session_dir,
        "df": None,
        "filename": None,
        "queries": [],     # list of user queries
        "answers": [],     # list of LLM answers (text)
        "plots": []        # list of plot file paths (in session dir)
    }

# --- Utilities to safely extract JSON from model output ---
def extract_json(text):
    """Try to find a JSON object in the model's reply and parse it."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None

# --- LLM call that requests structured JSON plan ---
def ask_model_for_plan(df, user_text):
    """
    Ask the LLM to return JSON only with either:
      { "action":"answer", "answer":"...optional textual answer..." }
    OR
      { "action":"plot", "plot": {"type":"hist|bar|line|scatter|box",
                                  "x": "<column-name or null>",
                                  "y": "<column-name or null>",
                                  "groupby": "<col or null>",
                                  "agg": "sum|mean|count|None",
                                  "title": "..." } }
    """
    # Build lightweight schema + sample
    cols = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in cols}
    sample = df.head(6).to_csv(index=False)

    system_msg = (
        "You are a strict JSON-outputting data assistant. "
        "Given the dataset schema and a user prompt, return ONLY a single JSON object (no markdown, no explanation). "
        "JSON must have either action='answer' with an 'answer' string, "
        "OR action='plot' with a 'plot' object describing type and columns to plot."
    )

    user_msg = (
        f"Columns: {cols}\nDtypes: {dtypes}\n"
        f"Sample rows (CSV):\n{sample}\n\n"
        f"User asks: {user_text}\n\n"
        "Return JSON only. Examples:\n"
        '{"action":"answer","answer":"Short answer..."}\n'
        '{"action":"plot","plot":{"type":"hist","x":"Age","title":"Age distribution"}}'
    )

    try:
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0
        )
        text = resp["choices"][0]["message"]["content"]
        plan = extract_json(text)
        return plan, text
    except Exception as e:
        return None, f"LLM error: {e}"

# --- High-level query handler ---
def handle_query(state, user_query):
    """
    Interprets user's natural query via LLM, stores answer in session memory.
    If LLM returns an 'answer' action, we show that as answer.
    If LLM returns a 'plot' action, we route to the plotting function (but we also accept 'answer' fallback).
    """
    if state["df"] is None:
        return state, "Please upload a CSV first."

    plan, raw_text = ask_model_for_plan(state["df"], user_query)

    # If model didn't produce JSON, fallback: ask plain answer (text) from model
    if plan is None:
        try:
            # fallback to a plain-text answer
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role":"system", "content":"You are a helpful data analyst."},
                    {"role":"user", "content": f"Columns: {list(state['df'].columns)}\nUser: {user_query}\nAnswer concisely."}
                ],
                temperature=0
            )
            answer = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            answer = f"LLM error: {e}"
        state["queries"].append(user_query)
        state["answers"].append(answer)
        return state, answer

    # if action == answer
    if plan.get("action") == "answer":
        ans = plan.get("answer") or "Model returned an empty answer."
        state["queries"].append(user_query)
        state["answers"].append(ans)
        return state, ans

    # if action == plot, delegate to plotter
    if plan.get("action") == "plot":
        plot_spec = plan.get("plot", {})
        # reuse plotting function to create plot file
        state, status, plot_path = create_plot_from_spec(state, plot_spec, user_query)
        # if plotting produced a file, return status and also show the path (Gradio Image handles it)
        return state, status

    # unknown action fallback
    err = "Model returned unknown action. Raw output:\n" + raw_text
    state["queries"].append(user_query)
    state["answers"].append(err)
    return state, err

# --- Plot creation (validate & execute) ---
def create_plot_from_spec(state, spec, user_query_context=""):
    """
    spec expected keys: type, x, y, groupby, agg, title
    Allowed types: hist, bar, line, scatter, box
    """
    df = state["df"]
    session_dir = state["dir"]

    ptype = (spec.get("type") or "hist").lower()
    x = spec.get("x")
    y = spec.get("y")
    groupby = spec.get("groupby")
    agg = (spec.get("agg") or "").lower() or None
    title = spec.get("title") or (f"{ptype} plot")

    # Validate columns
    for col in (c for c in [x, y, groupby] if c):
        if col not in df.columns:
            return state, f"Requested column '{col}' not found in dataset.", None

    fig, ax = plt.subplots(figsize=(8,5))
    try:
        if ptype == "hist":
            col = x or y
            if col is None:
                # pick first numeric column
                numeric = df.select_dtypes(include="number").columns.tolist()
                if not numeric:
                    return state, "No numeric columns available for histogram.", None
                col = numeric[0]
            df[col].dropna().astype(float).hist(ax=ax, bins=20)
            ax.set_title(title)

        elif ptype == "bar":
            if groupby and y:
                if agg in ("sum", "mean", "count"):
                    grp = getattr(df.groupby(groupby)[y], agg)()
                else:
                    grp = df.groupby(groupby)[y].sum()
                grp.plot(kind="bar", ax=ax)
                ax.set_title(title)
            elif x and y:
                df.groupby(x)[y].sum().plot(kind="bar", ax=ax)
                ax.set_title(title)
            else:
                return state, "Bar plot needs either (groupby + y) or (x + y).", None

        elif ptype == "line":
            if x and y:
                df.plot(x=x, y=y, kind="line", ax=ax)
                ax.set_title(title)
            else:
                return state, "Line plot requires both x and y columns.", None

        elif ptype == "scatter":
            if x and y:
                df.plot.scatter(x=x, y=y, ax=ax)
                ax.set_title(title)
            else:
                return state, "Scatter plot requires both x and y columns.", None

        elif ptype == "box":
            col = x or y
            if not col:
                numeric = df.select_dtypes(include="number").columns.tolist()
                if not numeric:
                    return state, "No numeric columns available for box plot.", None
                col = numeric[0]
            df.boxplot(column=col, ax=ax)
            ax.set_title(title)

        else:
            return state, f"Unsupported plot type '{ptype}'.", None

        # Save plot
        plot_index = len(state["plots"]) + 1
        plot_path = os.path.join(session_dir, f"plot_{plot_index}.png")
        plt.tight_layout()
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

        state["plots"].append(plot_path)
        # store query + a short descriptive answer for history
        desc = spec.get("title") or f"{ptype} plot ({x},{y})"
        state["queries"].append(user_query_context)
        state["answers"].append(f"Generated plot: {desc}")
        return state, f"Plot generated: {os.path.basename(plot_path)}", plot_path

    except Exception as e:
        plt.close(fig)
        return state, f"Plotting error: {e}", None

# --- Upload CSV handler ---
def upload_csv(state, file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return state, f"Failed to read CSV: {e}"

    # Place a copy in the session dir
    session_dir = state["dir"]
    saved_path = os.path.join(session_dir, "uploaded_data.csv")
    try:
        df.to_csv(saved_path, index=False)
    except Exception:
        # if write fails, keep going (not critical)
        pass

    state["df"] = df
    state["filename"] = os.path.basename(file_path)
    return state, f"Uploaded '{state['filename']}' ({df.shape[0]} rows, {df.shape[1]} cols)."

# --- Download (ZIP) handler ---
def download_all_results(state):
    """
    Creates a zip with:
      - uploaded_data.csv
      - queries_answers.txt
      - plots/* (all generated plots)
    Returns path to ZIP for Gradio to serve.
    The ZIP is scheduled for deletion after a short delay.
    """
    if state["df"] is None:
        return None

    session_dir = state["dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"results_{state['id']}_{timestamp}.zip"
    zip_path = os.path.join(session_dir, zip_name)

    try:
        # Ensure uploaded csv exists; create if not
        csv_path = os.path.join(session_dir, "uploaded_data.csv")
        if not os.path.exists(csv_path):
            state["df"].to_csv(csv_path, index=False)

        # write queries/answers
        qa_path = os.path.join(session_dir, "queries_answers.txt")
        with open(qa_path, "w", encoding="utf-8") as f:
            for q, a in zip(state["queries"], state["answers"]):
                f.write("Q: " + (q or "<empty>") + "\n")
                f.write("A: " + (a or "<empty>") + "\n")
                f.write("\n---\n\n")

        # create zip
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(csv_path, arcname="uploaded_data.csv")
            z.write(qa_path, arcname="queries_answers.txt")
            # add plots
            for i, p in enumerate(state["plots"], start=1):
                if os.path.exists(p):
                    z.write(p, arcname=os.path.join("plots", f"plot_{i}.png"))

        # schedule zip deletion (so it's removed after being downloaded)
        schedule_delete(zip_path, delay=ZIP_DELETE_DELAY)
        return zip_path

    except Exception as e:
        return None

# --- Cleanup session (remove all files for this session) ---
def end_session_and_cleanup(state):
    session_dir = state.get("dir")
    try:
        if session_dir and os.path.exists(session_dir):
            shutil.rmtree(session_dir)
    except Exception:
        pass
    # reinit state
    new_state = init_state()
    return new_state, "Session cleaned up."

# --- Gradio UI wiring ---
with gr.Blocks() as demo:
    gr.Markdown("# 📊 AI CSV Analyst (session memory + plots + LLM)")

    state = gr.State(init_state())

    with gr.Row():
        file_input = gr.File(label="Upload CSV (single file)", file_types=[".csv"], type="filepath")
        upload_btn = gr.Button("Upload")
        upload_status = gr.Textbox(label="Upload status", interactive=False)

    gr.Markdown("### Ask questions about the dataset")
    with gr.Row():
        query_in = gr.Textbox(placeholder="e.g., Which columns correlate with price? Or, give a short summary.", label="Natural language query")
        query_btn = gr.Button("Ask")
    query_out = gr.Textbox(label="LLM answer / status", lines=4)

    gr.Markdown("### Request plots")
    with gr.Row():
        plot_in = gr.Textbox(placeholder='e.g., "Plot histogram of alcohol" or "Plot mean quality by wine type".', label="Plot request (natural language)")
        plot_btn = gr.Button("Generate Plot")
    plot_status = gr.Textbox(label="Plot status", lines=2)
    plot_img = gr.Image(label="Last generated plot", type="filepath")

    with gr.Row():
        download_btn = gr.Button("Download all results (ZIP)")
        download_file = gr.File(label="Download ZIP")

        cleanup_btn = gr.Button("End session & cleanup")
        cleanup_out = gr.Textbox(label="Cleanup status")

    # Wiring with function outputs: first output is always the (updated) state
    upload_btn.click(upload_csv, inputs=[state, file_input], outputs=[state, upload_status])
    query_btn.click(handle_query, inputs=[state, query_in], outputs=[state, query_out])
    plot_btn.click(lambda st, txt: create_plot_from_spec(st, ask_model_for_plan(st["df"], txt)[0].get("plot") if ask_model_for_plan(st["df"], txt)[0] else {"type":"hist","x":None}, txt) if st["df"] is not None else (st, "Upload CSV first.", None),
                   inputs=[state, plot_in],
                   outputs=[state, plot_status, plot_img])
    # The above is a compact wiring: it uses the plan parse + plot creation. If the LLM fails it will try to fall back.
    download_btn.click(download_all_results, inputs=[state], outputs=[download_file])
    cleanup_btn.click(end_session_and_cleanup, inputs=[state], outputs=[state, cleanup_out])

if __name__ == "__main__":
    demo.launch()
