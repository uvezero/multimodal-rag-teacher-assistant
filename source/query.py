import chromadb
import numpy as np
import os
import gradio as gr
from PIL import Image
import ollama
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import sys
base_path = '/.../multimodal-search'
sys.path.append(base_path)
ollama.pull('llama3.2-vision')



# RAG Context Retrieval
def context_rag(query_text, query_img, db_path, collection_n, n_results=20):
    """
    Retrieves the context for the RAG model based on the query text and image.
    
    Parameters: 
        query_text (str): The query text.
        query_img (str): The path to the query image.
        db_path (str): The path to the ChromaDB database.
        colllection_n: Chromadb collection.
        n_results(int): Number of text results to retrieve.
        
    Returns:
        context_txt (dict): The context retrieved from the database for the query text.
        image_paths (list): The paths to the relevant images retrieved from the database.
    """
    #Client and Collection (ChromaDB does not recommend using PersistentClient,  however I will use it in this demo.)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_n)
    
    # FUTURE IMPLEMENTATION: Add metadata to database and context
    multimodal_ef = OpenCLIPEmbeddingFunction() # OpenCLIP embedding function
    imageloader = ImageLoader() # Image loader
    collection = client.get_or_create_collection(
        name=collection_n,
        embedding_function=multimodal_ef,
        data_loader=imageloader)  
        
    # Perform multimodal query
    if query_text:
        query_texts = [query_text] if query_text else None
        context_txt = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=["documents", "uris"]
        )  
    if query_img:
        image = Image.open(query_img)
        image_array = np.array(image)
        image_paths = []
        context_img = collection.query(query_images=[image_array], n_results=1, include=["documents", "uris"])
    else:
        print("No query provided.")
        
    # Extract image paths
    for uri in context_img.get("uris", [[]])[0]:
        if uri is not None:
            image_path = uri.replace("file://", "")
            image_paths.append(image_path)
            
    return context_txt, image_paths


def prompt_gen(query_text, query_img_path, context, image_paths, student):
    """
    Generates the prompt for the LLaMA model based on the query text, context, and student name.
    """
    # Context storage
    text_context = ""
    image_context = ""
    
    # Add text context
    for i, (document, uri) in enumerate(zip(context['documents'][0], context['uris'][0])):
        if uri is None:
            text_context = text_context + "**Snippet:** " + document + "\n\n" # Format the text context
            
    # Add image context
    if image_paths:
        for image_path in image_paths:
            image_context += f"![Relevant Image]({image_path})\n\n" # Format the image context
    
    # Prompt formatting        
    user_image_info = f"User's query image: {query_img_path}\n" if query_img_path else ""
    
    # construct prompt
    prompt = f"""
        You are a knowledgeable assistant. Your goal is to help the teacher create a personalized learning experience for the student {student}. Use the following information about the student or the question (remeber any relevant information mentioned is on the {student} notes) asked to adapt your response:


        **Query:** "{query_text}"
        
        **User's Query Image:**
        {user_image_info}
        
        **Student Name:** {student}

        **Relevant Information:**
        {text_context}
        {image_context}

        **Please provide your answer below in Markdown:**
        """
    return prompt


def stream_chat(message, query_img_path, history, db_path, student):
    """
    Streams the response from the Ollama model and sends it to the Gradio UI.
    
    Parameters:
        message (str): The user input message.
        history (list): A list of previous conversation messages.
        
    Returns:
        str: The chatbot's response chunk by chunk.
    """
    print(f"🔍 **Querying database for context**...")
    # context retrieval
    collection_n = student + "_collection"
    context, image_paths = context_rag(message, query_img_path, db_path, collection_n, 5)
    print(f"\n🔍 **Context Retrieval Complete**")
    # construct prompt
    prompt = prompt_gen(message, query_img_path, context, image_paths, student)
    
    print(f"\n🧠 **Sending prompt to LLaMA 3.2 Vision...**\n")
    print(prompt)  # Print the prompt to see what's being sent
    
    # append the user message to the conversation history
    if 'history' not in locals():
        history = []
    
    
    if image_paths:
        history.append({"role": "user", "content": prompt, "images": [image for image in image_paths]})
    else:
        history.append({"role": "user", "content": prompt})
    
    # initialize streaming from Ollama
    stream = ollama.chat(
        model='llama3.2-vision',
        messages=history,  # Full chat history including the current user message
        stream=True,
    )
    
    response_text = ""
    for chunk in stream:
        content = chunk['message']['content']
        response_text += content
        # yield response_text  # Send the response incrementally to the UI

    # append the assistant's full response to the history
    history.append({"role": "assistant", "content": response_text})
    print("\n\n✅ **LLaMA Response Complete**")
    return response_text, history