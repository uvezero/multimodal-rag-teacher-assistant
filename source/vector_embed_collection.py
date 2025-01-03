import sys
from semantic_router.encoders import HuggingFaceEncoder
import os
import ollama
import json
from pathlib import Path
from semantic_router.splitters import RollingWindowSplitter
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from semantic_router.utils.logger import logger
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings

base_path = '/home/juan/Documents/python/Portfolio projects/multimodal-search'
sys.path.append(base_path)

from utils.functions_preprocess import extract_text, extract_images, chunk_text


class embed():
    def __init__(self, db_path: str, path_imgs: str, collection_name: str):
        self.embedding_f = OpenCLIPEmbeddingFunction()
        self.encoder = HuggingFaceEncoder()
        self.splitter = RollingWindowSplitter(encoder=self.encoder, min_split_tokens=25, max_split_tokens=77, window_size=4, plot_splits=False, enable_statistics=True)
        self.imageloader = ImageLoader()
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_f, data_loader=self.imageloader)
        self.images_dir = path_imgs
        
    def embed_text(self, pdf_path: str) -> Embeddings:
        # Extract text from pdf
        text = extract_text(pdf_path)
        
        logger.setLevel('WARNING')
        self.encoder.score_threshold = 0.2
        # Split text into chunks
        chunks = chunk_text(text, pdf_path, self.splitter, save=False)
        
        # Generate embeddings for each chunk and store them in collection
        self.generate_txt_embeddings(chunks)
    
    def generate_txt_embeddings(self, chunks: list) -> Embeddings:
        documents_txt = []
        for content in chunks:
            section = content['file'] + ": "
            # clip's 256 sequence limit
            text = section + content['text_chunk'][:256-len(section)]
            documents_txt.append(text)
        document_ids = list(map(lambda tup: f"txt{tup[0]}", enumerate(documents_txt)))
        self.collection.add(documents=documents_txt, ids=document_ids)

    def generate_img_embeddings(self, images: list) -> Embeddings:
        image_uris = sorted(images)
        ids = [str(i) for i in range(len(image_uris))]
        self.collection.add(ids=ids, uris=image_uris)

    def embed_image(self, pdf_path: str) -> Embeddings:
        # Create images directory if not exists
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Extract images from pdf
        images = extract_images(pdf_path, self.images_dir)
        
        # Generate embeddings for each image and store them in collection
        self.generate_img_embeddings(images)

def test_all_good(db_path, collection_n):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_n)
    total_documents = collection.count()
    print(f"Total number of documents in the collection: {total_documents}")


john_report_path = '/.../john_test_data/john-report.pdf'
john_chem_path = '/.../john_test_data/Chemistry1-Revision-Guide-2018.pdf'
db_path = '/.../chromadb_store'
john_path_imgs = '/.../images_test/john_imgs'

collection_n = 'john_collection'

embed = embed(db_path, john_path_imgs, collection_n)

# Report
embed.embed_text(john_report_path)

#Chem Notes
embed.embed_text(john_chem_path)
embed.embed_image(john_chem_path)
