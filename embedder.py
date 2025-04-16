# embedder.py

import os
import json
import time
import fitz  # PyMuPDF
import faiss
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

# --- CONFIG ---
WATCH_DIR = "./files"
FAISS_INDEX_PATH = "./faiss.index"
CHUNKS_JSON_PATH = "./chunks.json"
VECTORS_NPY_PATH = "./vectors.npy"
PROCESSED_FILES_JSON = "processed_files.json"
COLLECTION_NAME = "rag_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 50

# --- Setup ---
os.makedirs(WATCH_DIR, exist_ok=True)
model = SentenceTransformer(EMBED_MODEL)
client = MilvusClient("milvus_lite.db")

# --- Milvus Setup ---
def create_milvus_collection():
    if COLLECTION_NAME in client.list_collections():
        print("[INFO] Collection already exists.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=384  # otomatis membuat field vector bernama 'embedding'
    )
    print("[SUCCESS] Collection created.")

def drop_milvus_collection():
    if COLLECTION_NAME in client.list_collections():
        client.drop_collection(COLLECTION_NAME)



create_milvus_collection()

# --- Helpers ---
def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    return "\n".join([page.get_text() for page in doc])

def embed_chunks(chunks):
    return model.encode(chunks).astype("float32")

def load_processed_files():
    return set(json.load(open(PROCESSED_FILES_JSON))) if os.path.exists(PROCESSED_FILES_JSON) else set()

def save_processed_files(files):
    with open(PROCESSED_FILES_JSON, "w") as f:
        json.dump(list(files), f)

def save_faiss_index(index, vectors, chunks):
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(VECTORS_NPY_PATH, vectors)
    with open(CHUNKS_JSON_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

# --- Main handler ---
class PDFHandler(FileSystemEventHandler):
    def __init__(self):
        self.processed = load_processed_files()
        self.index = faiss.IndexFlatL2(384)
        self.all_vectors = []
        self.all_chunks = []

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return

        filepath = event.src_path
        filename = os.path.basename(filepath)

        if filename in self.processed:
            print(f"[SKIP] Already processed: {filename}")
            return

        print(f"[NEW] Processing {filename}")
        text = extract_text_from_pdf(filepath)
        chunks = split_text(text)
        vectors = embed_chunks(chunks)

        # Store to FAISS
        self.index.add(vectors)
        self.all_vectors.extend(vectors)
        self.all_chunks.extend([{"filename": filename, "chunk": c} for c in chunks])

        # Store to Milvus Lite
        # Store to Milvus Lite with metadata
        data = [
            {
                "id": int(time.time() * 1000) + i,  # simple unique ID
                "vector": vectors[i],
                "text": chunks[i],
                "subject": filename,  # use filename as 'subject'
            }
            for i in range(len(chunks))
        ]

        client.insert(
            collection_name=COLLECTION_NAME,
            data=data
        )

        self.processed.add(filename)
        save_processed_files(self.processed)
        save_faiss_index(self.index, np.array(self.all_vectors), self.all_chunks)

        print(f"[DONE] {filename} indexed and stored.")

# --- Run Watcher ---
if __name__ == "__main__":
    print(f"[WATCHING] Folder for PDFs... Drop files into: {WATCH_DIR}/")
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()