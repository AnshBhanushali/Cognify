from typing import List
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
audio_collection = client.get_or_create_collection("audio_labels")

def save_audio_label_to_chroma(audio_id: str, label: str, embedding: List[float]):
    audio_collection.add(
        ids=[audio_id],
        embeddings=[embedding],
        documents=[label],
        metadatas=[{"source": "user_audio"}],
    )

def query_audio_label(embedding: List[float], n_results: int = 1):
    return audio_collection.query(query_embeddings=[embedding], n_results=n_results)
