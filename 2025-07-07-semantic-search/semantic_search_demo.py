import argparse
import re
import uuid
from typing import List
from InstructorEmbedding import INSTRUCTOR
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# -----------------------------
# Class: TextChunker
# -----------------------------
# Splits raw text into semantically meaningful chunks based on paragraph breaks.
class TextChunker:
    def __init__(self, min_length: int = 30):
        self.min_length = min_length

    def chunk(self, text: str) -> List[str]:
        # Split text on double line breaks (paragraphs)
        paragraphs = re.split(r"\n{2,}", text)
        # Return paragraphs that are longer than the minimum length
        return [p.strip() for p in paragraphs if len(p.strip()) > self.min_length]


# -----------------------------
# Class: Embedder
# -----------------------------
# Uses Instructor-XL to convert text into dense vector embeddings.
class Embedder:
    def __init__(self):
        # Load Instructor-XL embedding model
        self.model = INSTRUCTOR("hkunlp/instructor-xl")
        self.instruction = "Represent the document for retrieval:"

    def encode(self, texts: List[str]) -> List[List[float]]:
        # Apply the instruction prompt and encode each text to a vector
        return self.model.encode([[self.instruction, t] for t in texts])


# -----------------------------
# Class: VectorStore
# -----------------------------
# Manages Qdrant connection, collection setup, storage, and search.
class VectorStore:
    def __init__(self, collection_name: str = "semantic_demo"):
        self.collection_name = collection_name
        self.client = QdrantClient(host="localhost", port=6333)
        self._ensure_collection()

    def _ensure_collection(self):
        # Create the collection in Qdrant if it doesn't exist
        if self.collection_name not in self.client.get_collections().collections:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

    def add_documents(self, texts: List[str], vectors: List[List[float]]):
        # Prepare each document and its vector as a Qdrant PointStruct
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": text})
            for text, vec in zip(texts, vectors)
        ]
        # Upload points to Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query_vector: List[float], top_k: int = 5) -> List[str]:
        # Perform similarity search using the provided query vector
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        # Extract and return the matched text payloads
        return [hit.payload["text"] for hit in results]


# -----------------------------
# Class: SemanticSearchPipeline
# -----------------------------
# Orchestrates chunking, embedding, storing, and querying.
class SemanticSearchPipeline:
    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.store = VectorStore()

    def process_file(self, filepath: str):
        # Read the input file
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        # Chunk the text
        chunks = self.chunker.chunk(text)
        # Embed the chunks
        vectors = self.embedder.encode(chunks)
        # Store the chunks and vectors in Qdrant
        self.store.add_documents(chunks, vectors)
        print(f"Indexed {len(chunks)} chunks from {filepath}")

    def search(self, query: str, top_k: int = 5):
        # Embed the input query
        query_vector = self.embedder.encode([query])[0]
        # Retrieve top-K matching documents
        results = self.store.query(query_vector, top_k)
        print("\nTop Matches:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}\n")


# -----------------------------
# Main CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to input .txt file")
    parser.add_argument("--query", type=str, help="Query string for semantic search")
    args = parser.parse_args()

    # Initialize the semantic search pipeline
    pipeline = SemanticSearchPipeline()

    # If a file is provided, ingest and index it
    if args.file:
        pipeline.process_file(args.file)

    # If a query is provided, perform semantic search
    if args.query:
        pipeline.search(args.query)
