import os 
import fitz
import json
import numpy as np
import pymupdf
from io import BytesIO
from PIL import Image
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.encoders import HuggingFaceEncoder
from pathlib import Path
from semantic_router.splitters import RollingWindowSplitter
from semantic_router.utils.logger import logger

def recoverpix(doc, item):
    """
    ref: https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/examples/extract-images/extract-from-pages.py

    Recover an image from a PDF document, including handling SMask for transparency.
    """
    xref = item[0]  # xref of the image
    smask = item[1]  # SMask xref, if available

    # Case 1: Handle the image mask (SMask)
    if smask > 0:
        # Extract the base image
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.alpha:  # If there is an alpha channel, remove it
            pix0 = fitz.Pixmap(pix0, 0)

        # Extract the mask image
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])

        # Try combining the base image with the mask
        try:
            pix = fitz.Pixmap(pix0, mask)
        except Exception:
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])

        # Determine the image extension based on the number of color channels
        ext = "png" if pix.n <= 3 else "pam"

        return {
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext)
        }

    # Case 2: Handle ColorSpace definition (convert to RGB PNG)
    if "/ColorSpace" in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)
        return {
            "ext": "png",
            "colorspace": 3,
            "image": pix.tobytes("png")
        }

    # Default case: extract the image as is
    return doc.extract_image(xref)



def extract_text(pdf_path, output_file=None):
    """
    Extract text and images from a PDF file page by page, handling SMask and color space conversions.
    Organize the output into a structured format with text and image paths.
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)
    # Loop through each page of the document
    doc_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        text = page.get_text("text")
        text = text.replace('\n', ' ')
        doc_text.append(text)
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(doc_text, f, indent=4)
        print(f"Saved text to {output_file}")
    return doc_text

def convert_image(image, images_folder, xref, img_index):
    if image['ext'] not in ['jpg', 'jpeg', 'png']:
        img = Image.open(BytesIO(image['image']))
        img = img.convert("RGB")
        img_filename = os.path.join(images_folder, f"img_{xref}_{img_index}.jpg")
        img.save(img_filename, format="JPEG")
        image['ext'] = 'jpg'
        image['image'] = open(img_filename, "rb").read()
        return image

def extract_images(pdf_path, image_dir):
    doc = fitz.open(pdf_path)
    images = []
    imagescontent=[]
    # Loop through each page of the document
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images_on_page = page.get_images(full=True)
        
        
        if images_on_page:
            print(f"Extracting images from page {page_num + 1}...")
            for img_index, img in enumerate(images_on_page):
                try:
                    xref = img[0]  # Reference number for the image
                    # Recover the image using recoverpix
                    image = recoverpix(doc, img)
                    # Save the image to the page-specific directory
                    images_folder = os.path.join(image_dir, f"images")
                    os.makedirs(images_folder, exist_ok=True)
                    convert_image(image, images_folder, xref, img_index)
                    img_filename = os.path.join(images_folder, f"img_{xref}_{img_index}.{image['ext']}")
                    imagescontent.append({
                        "file_name": f"{os.path.splitext(os.path.basename(pdf_path))[0]}",
                        "image_path": os.path.abspath(img_filename)
                    })
                    with open(img_filename, "wb") as img_file:
                        img_file.write(image["image"])
                    print(f"Saved image {img_filename}")
                except Exception as e:
                    print(f"Error extracting image {img_index + 1} on page {page_num + 1}: {e}")
                    continue  # Skip to the next image or page
        else:
            print(f"No images found on page {page_num + 1}")
    
    # Save the image paths and captions to a JSON file
    image_paths_file = os.path.join(image_dir, "image_content.json")
    with open(image_paths_file, "w") as f:
        json.dump(imagescontent, f, indent=4)
    print(f"Saved image paths and captions to {image_paths_file}")

    return [i['image_path'] for i in imagescontent]



def serialize_document_split(chunks):
    """
    Converts a list of tuples in `text_chunk` into a structured dictionary format.
    """
    serialized_chunks = {}

    for key, value in chunks:
        if key == 'docs':
            # Flatten the 'docs' list into a single string
            serialized_chunks['text'] = " ".join(value)
        else:
            # Add all other keys to the current chunk
            serialized_chunks[key] = value

    return serialized_chunks


def chunk_text(json_file, pdf_path, splitter, save=True):
   
    
    chunked_text = []
    if isinstance(json_file, str):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json_file
    
    # Process the data (chunking logic here)
    chunks = []
    for page in data:
        splits = splitter([page])
        print(page)
        print('----------------------------------------------')
        for chunk in splits:
            docs_entry = [v for k, v in chunk if k == 'docs']
            if docs_entry:
                text_content = " ".join(docs_entry[0])
                # Filter out if it's too short
                if len(text_content.strip()) < 5:
                    continue
            chunks.append({
                "text_chunk": serialize_document_split(chunk)['text'],
                "file": f"{os.path.splitext(os.path.basename(pdf_path))[0]}"})
    if save is True:
        output_file = pdf_path.with_name('chunks.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)
    else:
        return chunks