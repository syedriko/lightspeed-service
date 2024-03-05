"""Utility script to generate embeddings."""

import argparse
import json
import os
import time

import faiss
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

PRODUCT_INDEX = "product"

if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description="embedding cli for task execution")
    parser.add_argument("-f", "--folder", help="Plain text folder path")
    parser.add_argument(
        "-md",
        "--model-dir",
        default="embeddings_model",
        help="Directory containing the embedding model",
    )
    parser.add_argument(
        "-mn",
        "--model-name",
        default="BAAI/bge-base-en",
        help="HF repo id of the embedding model",
    )
    parser.add_argument(
        "-c", "--chunk", type=int, default="1500", help="Chunk size for embedding"
    )
    parser.add_argument(
        "-l", "--overlap", type=int, default="10", help="Chunk overlap for embedding"
    )
    parser.add_argument("-o", "--output", help="Vector DB output folder")
    parser.add_argument(
        "-s", "--faiss-vector-size", type=int, default="768", help="Faiss vector size"
    )
    args = parser.parse_args()

    PERSIST_FOLDER = args.output

    faiss_index = faiss.IndexFlatL2(args.faiss_vector_size)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    os.environ["HF_HOME"] = args.model_dir
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    Settings.chunk_size = args.chunk
    Settings.chunk_overlap = args.overlap
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.model_dir)
    Settings.llm = None

    documents = SimpleDirectoryReader(args.folder, recursive=True).load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    index.set_index_id(PRODUCT_INDEX)
    index.storage_context.persist(persist_dir=PERSIST_FOLDER)

    metadata = {}
    metadata["execution-time"] = time.time() - start_time
    metadata["llm"] = "None"
    metadata["embedding-model"] = args.model_name
    metadata["index_id"] = PRODUCT_INDEX
    metadata["vector-db"] = "faiss"
    metadata["chunk"] = args.chunk
    metadata["overlap"] = args.overlap
    metadata["total-embedded-files"] = len(documents)

    with open(os.path.join(PERSIST_FOLDER, "metadata.json"), "w") as file:
        file.write(json.dumps(metadata))
