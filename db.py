from typing import Set
import os
from config import Config
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import logging
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
import requests
import ray
import chainlit as cl
from promp_template import ContextTemplate

# Configure logging at the start of your project
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    filename='database.log',  # Save logs to a file (optional)
    filemode='w'  # Append mode ('w' to overwrite)
)
logger = logging.getLogger(__name__)

def hash_lib_pdf(file_path: str) -> str:
    """Hash the PDF file to create a unique identifier"""
    import hashlib
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# @ray.remote
class VectorDatabase:
    def __init__(
        self,
        qdrant_url: str,
        embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        self.late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        self.client = QdrantClient(url=qdrant_url)
        self.embed_model_id = embed_model_id
        self.url = qdrant_url

    def create_collection(self, collection_name: str) -> None:
        """Create a new collection in Qdrant"""
        if self._check_collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists in Qdrant")
        else:
            output = self.client.create_collection(
            collection_name,
            vectors_config={
                "all-MiniLM-L6-v2": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
                "colbertv2.0": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    )
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    )
                }
            )
            logger.info(f"Created collection '{collection_name}' in Qdrant ({output})")
        
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from Qdrant"""
        if not self._check_collection_exists():
            logger.info(f"Collection '{collection_name}' does not exist in Qdrant")
        else:
            output = self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}' from Qdrant ({output})")

    def _check_collection_exists(self, collection_name:str) -> bool:
        """Check if the collection already exists in Qdrant"""
        return self.client.collection_exists(collection_name)

    def _get_existing_hashes(self, collection_name:str) -> Set[str]:
        """Retrieve existing document hashes from the collection"""
        hashes = set()
        records, _ = self.client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        for record in records:
            if record.payload["metadata"].get("binary_hash"):
                hashes.add(record.payload["metadata"]["binary_hash"]) # Different chunking strategies may have different hash keys
            elif record.payload.get("binary_hash"):
                hashes.add(record.payload["binary_hash"])
            else:
                hashes.add(record.payload["metadata"]["dl_meta"]["origin"]["binary_hash"])
        return hashes

    def add_documents(self, file: cl.File, collection: str) -> None:
        file_path = file.path
        # Check if the collection exists
        if not self._check_collection_exists(collection):
            logger.error(f"Collection '{collection}' does not exist in Qdrant. Please create the collection first.")
            raise ValueError(f"Collection '{collection}' does not exist in Qdrant. Please create the collection first.")
        file_hash = hash_lib_pdf(file_path)
        
        # Check if the file already exists in the collection
        existing_hashes = self._get_existing_hashes(collection)
        if file_hash in existing_hashes:
            logger.info(f"File '{file_path}' already exists in the collection '{collection}'. Updating the document.")
        
        # Send the PDF file to the server for processing
        url = f"{os.getenv('NOUGAT_URL')}/predict/"
        headers = {
            "accept": "application/json"
        }
        
        # Open the PDF file in binary mode.
        with open(file_path, "rb") as pdf_file:
            files = {
                "file": (file_path, pdf_file, "application/pdf")
            }
            response = requests.post(url, headers=headers, files=files)

        # Save the markdown content to a file.
        markdown = response.json()
        with open(f".files/{file_hash}.md", "w") as md_file:
            md_file.write(markdown)
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ]
        )
        splits = [split for split in splitter.split_text(markdown)]
        title = splits[0].metadata["Header_1"]
        # TODO: split data into smaller chunks (word based or semantic based), need reference to the original chunk
        # TODO: admin.collection
        
        import tqdm

        # Create and persist a Qdrant vector database from the chunked documents
        batch_size = 8
        def chunker(iterable, n):
            """Yield successive n-sized chunks from iterable."""
            for i in range(0, len(iterable), n):
                yield iterable[i:i + n]
                
        for b, batch in tqdm.tqdm(enumerate(chunker(splits, batch_size)), desc="Uploading documents"):
            text = [doc.page_content for doc in batch]
            metadata = [doc.metadata for doc in batch]
            dense_embeddings = list(self.dense_embedding_model.passage_embed(text))
            bm25_embeddings = list(self.bm25_embedding_model.passage_embed(text))
            late_interaction_embeddings = list(self.late_interaction_embedding_model.passage_embed(text))
            result = self.client.upsert(
                collection,
                points=[
                    models.PointStruct(
                        id=int(b*batch_size + i),
                        vector={
                            "all-MiniLM-L6-v2": dense_embeddings[i].tolist(),
                            "bm25": bm25_embeddings[i].as_object(),
                            "colbertv2.0": late_interaction_embeddings[i].tolist(),
                        },
                        payload={
                            "metadata": metadata[i],
                            "text": text[i],
                            "binary_hash": file_hash,
                            "file_name": file.name,
                            "file_path": file_path,
                            "title": title
                        }
                    )
                    for i, _ in enumerate(batch)
                ],
            )
        return result
    
    def context_retrieval(self, prompt:str, collection:str):
        """Retrieve context from the database."""
        query_text = [prompt]
        dense_query_vector = next(self.dense_embedding_model.query_embed(query_text))
        sparse_query_vector = next(self.bm25_embedding_model.query_embed(query_text))
        late_query_vector = next(self.late_interaction_embedding_model.query_embed(query_text))

        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using="all-MiniLM-L6-v2",
                limit=6,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query_vector.as_object()),
                using="bm25",
                limit=6,
            ),
            models.Prefetch(
                query=late_query_vector,
                using="colbertv2.0",
                limit=6,
            ),
        ]
        results = self.client.query_points(
                collection,
                prefetch=prefetch,
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF, #https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a
                ),
                with_payload=True,
                limit=1,
            )

        point_id = [point.id for point in results.points]
        scores = [point.score for point in results.points]
        points = self.client.retrieve(collection_name=collection, ids=point_id)
        context = []
        pdfs = set()
        for point, score in zip(points, scores):
            template = ContextTemplate()
            context.append(template.render({
                "score": score,
                "text": point.payload["text"],
                "title": point.payload["title"],
                "structure": point.payload["structure"]
            }))
            pdfs.add(point.payload["file_name"])
        return context, pdfs
    
# Usage example
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    collection_name = "admin.collection"
    vector_db = VectorDatabase(
        qdrant_url=os.getenv('QDRANT_URL'),
        embed_model_id=Config.EMBED_MODEL_ID
    )
    vector_db.create_collection(collection_name)
    import chainlit as cl
    class File: 
        def __init__(self, path: str, name: str):
            self.path = path
            self.name = name
    file = File(path="./data/2311-SMERF.04079v1.pdf", name="2311-SMERF.04079v1.pdf")
    vector_db.add_documents(
        file,
        collection=collection_name
    )

    ## Hybrid Search
    query_text = "explain Polyline Sequence Representation in SMERF?"
    context, pdfs = vector_db.context_retrieval(query_text, collection_name)
    print(context)
    print(pdfs)
    ## Ray Actor Usage Example
# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     load_dotenv()
#     ray.init()
    
#     collection_name = "admin.collection"
#     vector_db = VectorDatabase.remote(
#         qdrant_url=os.getenv('QDRANT_URL'),
#         embed_model_id=Config.EMBED_MODEL_ID
#     )
    
#     # Create collection
#     vector_db.create_collection.remote(collection_name)
    
#     # Add documents 
#     vector_db.add_documents.remote(
#         file_path="./data/2311-SMERF.04079v1.pdf",
#         collection=collection_name
#     )

#     # Hybrid Search
#     query_text = "explain Polyline Sequence Representation in SMERF?"
#     context, pdfs = ray.get(vector_db.context_retrieval.remote(query_text, collection_name))
#     print(context)
#     print(pdfs)
    
    # ray.shutdown()