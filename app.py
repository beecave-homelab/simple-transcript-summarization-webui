import os
import gradio as gr
from transformers import pipeline
import torch
import re

# Determine if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Function to create a unique filename
def create_unique_filename(base_name, directory, extension):
    counter = 1
    unique_name = f"{base_name}{extension}"
    while os.path.exists(os.path.join(directory, unique_name)):
        unique_name = f"{base_name}_{counter}{extension}"
        counter += 1
    return unique_name

# Function to create the summarization interface within a tab
def create_summarization_tab(model_name, model_description):
    # Load the specified model for summarization
    summarizer = pipeline("summarization", model=model_name, device=0 if device == "mps" else -1)

    # Ensure the summary folder exists
    if not os.path.exists("summary"):
        os.makedirs("summary")

    def chunk_text(text, max_chunk_size=1024):
        """Splits the text into chunks of max_chunk_size tokens."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_length += len(word) + 1  # Adding 1 for the space or punctuation
            if current_length > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = len(word) + 1
            current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        print(f"Total chunks created: {len(chunks)}")  # Debug: Print the number of chunks created
        return chunks

    def summarize(input_data):
        text_input = input_data.get("text", "")
        files = input_data.get("files", [])
        summary_filename = ""

        if files:
            # Use the uploaded file's name
            file_path = files[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            text_input = file_content if file_content else text_input
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            summary_filename = f"{base_name}_{model_description}"
        else:
            base_name = None

        if not text_input:
            return {"text": "Please provide some text or upload a file."}

        # Split text into chunks
        chunks = chunk_text(text_input, max_chunk_size=1024)

        # Summarize each chunk and combine results
        summaries = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"Processing chunk {i} of {len(chunks)}...")  # Debug: Print the current chunk being processed
            summary = summarizer(chunk, max_length=512, min_length=150, do_sample=False)[0]['summary_text']
            summaries.append(summary)

        final_summary = " ".join(summaries)

        # Generate filename if no file was uploaded
        if not base_name:
            # Use the first 5 words of the summary
            base_name = "_".join(re.findall(r'\w+', final_summary)[:5])
            summary_filename = f"{base_name}_{model_description}"

        # Ensure the filename is unique
        unique_filename = create_unique_filename(summary_filename, "summary", ".txt")

        # Save the summary to a file
        with open(os.path.join("summary", unique_filename), "w") as f:
            f.write(final_summary)

        return {"text": final_summary}

    with gr.Tab(model_description):
        gr.Markdown(f"## Summarization using {model_description}")
        multimodal_input = gr.MultimodalTextbox(
            label="Enter text or upload a file",
            placeholder="Enter the text to summarize here...",
            file_types=[".txt"],
            file_count="single",
            lines=10,
            max_lines=20
        )
        
        output_text = gr.Textbox(label="Summary", lines=5)

        # Use the send button from MultimodalTextbox
        multimodal_input.submit(summarize, inputs=multimodal_input, outputs=output_text)

# Gradio interface with Tabs
with gr.Blocks() as demo:
    with gr.Tabs():
        # Tab for mBART-50 model
        create_summarization_tab("facebook/mbart-large-50", "mBART-50")
        # Tab for mT5-large model
        create_summarization_tab("google/mt5-large", "mT5-Large")
        # Tab for mT5-small model
        create_summarization_tab("google/mt5-small", "mT5-Small")
        # Tab for Facebook/BART cnn model
        create_summarization_tab("facebook/bart-large-cnn", "BART-Large-CNN")

# Launch the app
demo.launch()