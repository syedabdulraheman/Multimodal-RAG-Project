"""
Multimodal RAG System - 100% Local (No API Key Required)
File: multimodal_rag.py

This is the backend module that can be imported by other scripts.
"""

import fitz  # PyMuPDF
import torch
import numpy as np
from PIL import Image
import io
import base64
from transformers import (
    CLIPProcessor, 
    CLIPModel,
    BlipProcessor,
    BlipForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import faiss
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: Initialize Local Models (No API Keys!)
# ============================================================================

def load_models():
    """Load all models - called once at startup"""
    print("Loading models locally... (first run may take time to download)")

    # CLIP for unified embeddings
    print("1/3 Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # BLIP for visual question answering
    print("2/3 Loading BLIP...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    blip_model.eval()

    # FLAN-T5 for text generation (lightweight but powerful)
    print("3/3 Loading FLAN-T5...")
    text_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    text_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    text_model.eval()

    print("✓ All models loaded successfully!\n")
    
    return clip_model, clip_processor, blip_model, blip_processor, text_model, text_tokenizer

# Load models globally (once)
clip_model, clip_processor, blip_model, blip_processor, text_model, text_tokenizer = load_models()

# ============================================================================
# PART 2: Embedding Functions
# ============================================================================

def embed_image(image_data):
    """Embed image using CLIP"""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data
    
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text):
    """Embed text using CLIP"""
    inputs = clip_processor(
        text=text, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# ============================================================================
# PART 3: PDF Processing
# ============================================================================

def process_pdf(pdf_path):
    """Extract text and images from PDF"""
    doc = fitz.open(pdf_path)
    all_docs = []
    all_embeddings = []
    image_store = {}
    
    for page_num, page in enumerate(doc):
        # Process text
        text = page.get_text()
        if text.strip():
            # Split into chunks
            chunks = [text[i:i+500] for i in range(0, len(text), 400)]
            
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.strip():
                    embedding = embed_text(chunk)
                    all_embeddings.append(embedding)
                    all_docs.append({
                        'content': chunk,
                        'type': 'text',
                        'page': page_num,
                        'chunk': chunk_idx
                    })
        
        # Process images
        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"page_{page_num}_img_{img_idx}"
                
                # Store image
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                image_store[image_id] = pil_image
                
                # Embed image
                embedding = embed_image(pil_image)
                all_embeddings.append(embedding)
                
                all_docs.append({
                    'content': f"[Image from page {page_num}]",
                    'type': 'image',
                    'page': page_num,
                    'image_id': image_id
                })
                
            except Exception as e:
                print(f"Error processing image {img_idx} on page {page_num}: {e}")
    
    doc.close()
    return all_docs, np.array(all_embeddings), image_store

# ============================================================================
# PART 4: Vector Store with FAISS
# ============================================================================

class MultimodalVectorStore:
    """Simple FAISS-based vector store"""
    
    def __init__(self, embeddings, documents):
        self.documents = documents
        self.dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine sim)
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings)
    
    def search(self, query_embedding, k=5):
        """Search for similar documents"""
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                'document': self.documents[idx],
                'score': float(score)
            })
        return results

# ============================================================================
# PART 5: Multimodal Question Answering
# ============================================================================

def answer_with_image(image, question):
    """Use BLIP to answer questions about an image"""
    inputs = blip_processor(image, question, return_tensors="pt")
    
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=50)
        answer = blip_processor.decode(out[0], skip_special_tokens=True)
    
    return answer

def generate_text_answer(context, question):
    """Use FLAN-T5 to generate answers from text"""
    # Create a clear prompt
    prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context[:800]}

Question: {question}

Answer:"""
    
    inputs = text_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = text_model.generate(
            **inputs,
            max_length=150,
            min_length=20,
            num_beams=4,
            temperature=0.7,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        answer = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # If answer is too short or empty, provide context-based answer
    if not answer or len(answer.strip()) < 10:
        # Provide a simple extraction from context
        sentences = context.split('.')[:3]
        answer = '. '.join(sentences) + '.'
    
    return answer.strip()

# ============================================================================
# PART 6: Main RAG Pipeline
# ============================================================================

def multimodal_rag(query, vector_store, image_store, k=3):
    """Complete multimodal RAG pipeline"""
    
    # Embed query
    query_embedding = embed_text(query)
    
    # Retrieve relevant documents
    results = vector_store.search(query_embedding, k=k)
    
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    print(f"Retrieved {len(results)} documents:\n")
    
    text_context = []
    image_answers = []
    
    for i, result in enumerate(results):
        doc = result['document']
        score = result['score']
        
        print(f"{i+1}. [{doc['type'].upper()}] Page {doc['page']} (Score: {score:.3f})")
        
        if doc['type'] == 'text':
            preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
            print(f"   {preview}\n")
            text_context.append(doc['content'])
            
        elif doc['type'] == 'image':
            print(f"   Image ID: {doc['image_id']}\n")
            image = image_store[doc['image_id']]
            img_answer = answer_with_image(image, query)
            image_answers.append(f"From image on page {doc['page']}: {img_answer}")
    
    # Combine contexts
    all_context = "\n\n".join(text_context)
    if image_answers:
        all_context += "\n\nImage analysis:\n" + "\n".join(image_answers)
    
    # Generate final answer
    print("Generating answer...\n")
    final_answer = generate_text_answer(all_context, query)
    
    print(f"{'='*70}")
    print(f"ANSWER:\n{final_answer}")
    print(f"{'='*70}\n")
    
    return final_answer

# ============================================================================
# PART 7: Main Execution
# ============================================================================

# ============================================================================
# PART 7: Main Execution
# ============================================================================

if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════
    # 📄 CHANGE YOUR PDF NAME HERE
    # ═══════════════════════════════════════════════════════════════════
    pdf_path = "multimodal_sample.pdf"  # ← PUT YOUR PDF NAME HERE
    # Example: pdf_path = "my_document.pdf"
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"Processing PDF: {pdf_path}\n")
    
    documents, embeddings, image_store = process_pdf(pdf_path)
    print(f"✓ Extracted {len(documents)} documents ({sum(1 for d in documents if d['type']=='text')} text, {sum(1 for d in documents if d['type']=='image')} images)\n")
    
    # Create vector store
    vector_store = MultimodalVectorStore(embeddings, documents)
    print("✓ Vector store created\n")
    
    # Example queries
    queries = [
        "What does the chart show about revenue trends?",
        "Summarize the main findings from the document",
        "What visual elements are present in the document?",
        "Which quarter had the highest revenue?"
    ]
    
    print("\n" + "="*70)
    print("MULTIMODAL RAG SYSTEM - READY")
    print("="*70)
    
    for query in queries:
        multimodal_rag(query, vector_store, image_store, k=3)
        print("\n")