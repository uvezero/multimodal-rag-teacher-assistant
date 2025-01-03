import os
import gradio as gr
import sys
import ollama
from query import stream_chat

base_path = '/.../multimodal-search'
sys.path.append(base_path)
ollama.pull('llama3.2-vision')

# Database and Collection (ChromaDB)
db_path = "/.../chromadb_store"

conversation_history = []

def stream_chat_ui(query_text, query_image=None, history=None, student=""):
    """
    Handles the student selection, query text, and optional query image.
    Uses the existing stream_chat function for context retrieval and LLM response.
    """
    global conversation_history

    # Save uploaded image (if any)
    query_img_path = None
    if query_image:
        query_img_path = "/tmp/uploaded_image.png"
        query_image.save(query_img_path)

    # Call your existing stream_chat function
    response_text, updated_history = stream_chat(query_text, query_img_path, history if history else [], db_path, student)

    # Update conversation history
    conversation_history.append((query_text, response_text))

    # Return updated conversation history for the chatbot
    return conversation_history

# Gradio Interface
with gr.Blocks() as app:
    with gr.Row():
        student_name = gr.Dropdown(
            label="Select Student",
            choices=["john", "jane"],
            value="john"
        )
    with gr.Row():
        chatbot = gr.Chatbot(scale=2, height=750, bubble_full_width=False)
    with gr.Row():
        text_box = gr.Textbox(label="Enter your query", placeholder="Type your question here...")
        image_box = gr.Image(type="pil", label="Upload an image (optional)")
        submit_button = gr.Button("Submit")

    # Handle button click
    submit_button.click(
        fn=stream_chat_ui,
        inputs=[text_box, image_box, chatbot, student_name],
        outputs=chatbot
    )

app.launch()