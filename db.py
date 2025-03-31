from typing import Optional, Set
from config import Config
from qdrant_client import QdrantClient, models
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from unstructured.partition.pdf import partition_pdf
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType, DoclingLoader
from docling.document_converter import DocumentConverter
import logging
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
import ollama
from langchain.prompts import ChatPromptTemplate
import requests
from langchain_core.output_parsers import StrOutputParser
import re
from openai import OpenAI
# Configure logging at the start of your project
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    filename='app.log',  # Save logs to a file (optional)
    filemode='a'  # Append mode ('w' to overwrite)
)
logger = logging.getLogger(__name__)
# llm  = OllamaLLM(model=Config.LLM)
# with open("temp.jpeg", "rb") as file:
#     base64_image = base64.b64encode(file.read()).decode("utf-8")
# response = ollama.generate(
#     model="gemma3:12b",
#     prompt="Write image caption please?",
#     images=[base64_image]
# )
# print(response)


def hash_lib_pdf(file_path: str) -> str:
    """Hash the PDF file to create a unique identifier"""
    import hashlib
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    # return hasher.hexdigest()
    return hashlib.md5(file_path.encode()).hexdigest()
    # return hashlib.sha256(file_path.encode()).hexdigest()
    # return hashlib.sha1(file_path.encode()).hexdigest()

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
            else:
                hashes.add(record.payload["metadata"]["dl_meta"]["origin"]["binary_hash"])
        return hashes

    def add_documents(self, file_path: str, collection: str, export_type: ExportType = ExportType.DOC_CHUNKS) -> None:
        # Check if the collection exists
        if not self._check_collection_exists(collection):
            logger.error(f"Collection '{collection}' does not exist in Qdrant. Please create the collection first.")
            return
        
        """Add new documents to the vector database"""
        
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        md_filename = "data/temp.md"
        result.document.save_as_markdown(md_filename)
        # Load and split documents
        loader = DoclingLoader(
            file_path=file_path,
            export_type=export_type,
            chunker=HybridChunker(tokenizer=self.embed_model_id),
        )
        splits = loader.load()

        # Filter new documents
        existing_hashes = self._get_existing_hashes(collection)
        new_docs = [
            doc for doc in splits 
            if doc.metadata["dl_meta"]["origin"]["binary_hash"] not in existing_hashes
        ]

        if not new_docs:
            logger.info("Documents {} already exist in the collection: {}".format(file_path, collection))
        else:
            import tqdm

            batch_size = 4
            def chunker(iterable, n):
                """Yield successive n-sized chunks from iterable."""
                for i in range(0, len(iterable), n):
                    yield iterable[i:i + n]
                    
            for b, batch in tqdm.tqdm(enumerate(chunker(new_docs, batch_size)), desc="Uploading documents"):
                text = [doc.page_content for doc in batch]
                metadata = [doc.metadata for doc in batch]
                dense_embeddings = list(self.dense_embedding_model.passage_embed(text))
                bm25_embeddings = list(self.bm25_embedding_model.passage_embed(text))
                late_interaction_embeddings = list(self.late_interaction_embedding_model.passage_embed(text))
                self.client.upload_points(
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
                                "text": text[i]
                            }
                        )
                        for i, _ in enumerate(batch)
                    ],
                    # We send a lot of embeddings at once, so it's best to reduce the batch size.
                    # Otherwise, we would have gigantic requests sent for each batch and we can
                    # easily reach the maximum size of a single request.
                    batch_size=batch_size,  
                )
            
            # output = QdrantVectorStore.from_documents(
            #     collection_name=collection,
            #     documents=new_docs,
            #     embedding=self.embed_model,
            # )
            logger.info(f"Added {file_path} new documents to the collection: {collection} ")
            
    def add_documents2(self, file_path: str, collection: str, export_type: ExportType = ExportType.DOC_CHUNKS) -> None:
        # Check if the collection exists
        if not self._check_collection_exists(collection):
            logger.error(f"Collection '{collection}' does not exist in Qdrant. Please create the collection first.")
            return
        file_hash = hash_lib_pdf(file_path)
        # Check if the file already exists in the collection
        existing_hashes = self._get_existing_hashes(collection)
        if file_hash in existing_hashes:
            logger.info(f"File '{file_path}' already exists in the collection '{collection}'. Skipping upload.")
            return
        """Add new documents to the vector database"""
        # Load and split documents
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,  # extract tables
            strategy="hi_res",
            extract_image_block_types=[
                "Image"
            ],  # Add 'Table' to list to extract image of tables
            # image_output_dir_path=output_path,   # if None, images and tables will saved in base64
            extract_image_block_to_payload=True,  # if true, will extract base64 for API usage
            chunking_strategy="by_title",  # or 'basic'
            max_characters=6000,  # defaults to 500
            combine_text_under_n_chars=1300,  # defaults to 0
            new_after_n_chars=4000,
        )
        tables = []
        texts = []
        table_texts = []
        figure_texts = []
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                texts.append(chunk.text)
                chunk_els = chunk.metadata.orig_elements
                for i, el in enumerate(chunk_els):
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
                    elif "Table" in str(type(el)):
                        tables.append(el.metadata.text_as_html)
                    elif "NarrativeText" in str(type(el)) or "FigureCaption" in str(type(el)):
                        if "Table" in el.text or "TABLE" in el.text or "Tab." in el.text:
                            table_texts.append(el.text)
                        elif "Figure" in el.text or "FIGURE" in el.text or "Fig." in el.text:
                            figure_texts.append(el.text)
            
        # text_summaries = texts
        # text_summaries = [Document(page_content=t.strip(), metadata={"binary_hash": file_hash, "text": t, "id_key": f"text_{i}"}) for i, t in enumerate(text_summaries)]
               
        prompt_text = """
        You are an assistant tasked with summarizing text.
        Give a concise summary of the text try not to lose architecture information.

        Respond only with the summary, no additionnal comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        Text chunk: {element}
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        # Summary chain
        model = OllamaLLM(model="gemma3:12b")
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
        text_summaries = [Document(page_content=t, metadata={"binary_hash": file_hash, "text": texts[i].strip(), "id_key": f"text_{i}"}) for i, t in enumerate(text_summaries)]
        response_figure = ollama.generate(
            model="gemma3:12b",
            prompt=f"""
                Task: Given a list of figures along with their captions as well as relevant text, return a list of unique figures with their captions.
                Figures:
                {figure_texts}
                Answer Format: Figure [1]: Caption 1 | Figure [2]: Caption 2
            """)
        
        response_table = ollama.generate(
            model="gemma3:12b",
            prompt=f"""
                Task: Given a list of tables along with their captions as well as relevant text, return a list of unique tables with their captions.
                Tables:
                {table_texts}
                Answer Format: Table [1]: Caption 1 | Table [2]: Caption 2
            """)

        table_caption_dict = {}
        parts = response_table.response.split('|')
        for part in parts:
            part_clean = part.strip()
            match = re.match(r'Table \[(\d+)\]:\s*(.*)', part_clean)
            if match:
                key = match.group(1)
                value = match.group(2)
                table_caption_dict[key] = "Table {}: {}".format(key, value)
                
        table_content_dict = {}
        for i, table in enumerate(tables):
            table_content_dict[f"{chr(65 + i)}"] = table
            
            
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        prompt_table=f"""Task: Given 2 lists. One list contains tables in HTML with format 'A': 'content', and one list contains captions with format '1': 'caption'. Try to match them together using table content and caption content.
                Answer only with this format without your reasoning: A:1|B:3|C:6...
                Tables:{table_content_dict},
                Captions: {table_caption_dict}"""
                
        completion_table = client.chat.completions.create(
            model="o1-mini",
            messages=[{
                "role": "user",
                "content": f'{prompt_table}'
            }]
        )
        table_summaries = []
        for i, pair in enumerate(completion_table.choices[0].message.content.split("|")):
            key, value = pair.split(":")
            table_content = table_content_dict[key.strip()]
            table_caption = table_caption_dict[value.strip()]
            table_summaries.append(Document(page_content=table_caption, metadata={"binary_hash": file_hash, "table_content": table_content, "id_key": f"table_{i}"}))
            
        figure_summaries = []
        for i, figure_text in enumerate(response_figure.response.split("|")):
            figure_text = figure_text.strip()
            figure_summaries.append(Document(page_content=figure_text, metadata={"binary_hash": file_hash, "image_base64": images_b64[i], "id_key": f"figure_{i}"}))

        # Create and persist a Qdrant vector database from the chunked documents
        import tqdm
        new_docs = text_summaries + table_summaries + figure_summaries
        batch_size = 4
        def chunker(iterable, n):
            """Yield successive n-sized chunks from iterable."""
            for i in range(0, len(iterable), n):
                yield iterable[i:i + n]
                
        for b, batch in tqdm.tqdm(enumerate(chunker(new_docs, batch_size)), desc="Uploading documents"):
            text = [doc.page_content for doc in batch]
            metadata = [doc.metadata for doc in batch]
            dense_embeddings = list(self.dense_embedding_model.passage_embed(text))
            bm25_embeddings = list(self.bm25_embedding_model.passage_embed(text))
            late_interaction_embeddings = list(self.late_interaction_embedding_model.passage_embed(text))
            self.client.upload_points(
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
                            "text": text[i]
                        }
                    )
                    for i, _ in enumerate(batch)
                ],
                # We send a lot of embeddings at once, so it's best to reduce the batch size.
                # Otherwise, we would have gigantic requests sent for each batch and we can
                # easily reach the maximum size of a single request.
                batch_size=batch_size,  
            )
        
        # output = QdrantVectorStore.from_documents(
        #     collection_name=collection,
        #     documents=new_docs,
        #     embedding=self.embed_model,
        # )
        logger.info(f"Added {file_path} new documents to the collection: {collection} ")

    def add_documents3(self, file_path: str, collection: str, export_type: ExportType = ExportType.DOC_CHUNKS) -> None:
        url = "http://127.0.0.1:8503/predict/"
        headers = {
            "accept": "application/json"
        }
        # Open the PDF file in binary mode.
        with open(file_path, "rb") as pdf_file:
            files = {
                "file": (file_path, pdf_file, "application/pdf")
            }
            response = requests.post(url, headers=headers, files=files)

        # Print the JSON response from the server.
        print(response.json())
        markdown = response.json()
        # Save the markdown content to a file.
        with open("output.md", "w") as md_file:
            md_file.write(markdown)
        return 
        
# Usage example
if __name__ == "__main__":
    # from dotenv import load_dotenv
    # load_dotenv()

    vector_db = VectorDatabase(
        qdrant_url=Config.QDRANT_URL,
        embed_model_id=Config.EMBED_MODEL_ID
    )
    vector_db.create_collection("rag2")
    vector_db.add_documents3(
        file_path="./data/2311-SMERF.04079v1.pdf",
        collection="rag2",
        export_type=ExportType.DOC_CHUNKS
    )
    # vector_db.add_documents2(
    #     file_path="./data/2311-SMERF.04079v1.pdf",
    #     collection="rag2",
    #     export_type=ExportType.DOC_CHUNKS
    # )
    # vector_db.add_documents(
    #     file_path="./data/2311-SMERF.04079v1.pdf",
    #     collection="rag2",
    #     export_type=ExportType.DOC_CHUNKS
    # )
    
    ## Hybrid Search
    query_text = ["explain Polyline Sequence Representation in SMERF?"]
    dense_query_vector = next(vector_db.dense_embedding_model.query_embed(query_text))
    sparse_query_vector = next(vector_db.bm25_embedding_model.query_embed(query_text))
    late_query_vector = next(vector_db.late_interaction_embedding_model.query_embed(query_text))
    # vector_db.client.query_points(
    #     "rag2",
    #     query=next(vector_db.dense_embedding_model.query_embed(query_text)),
    #     using="all-MiniLM-L6-v2",
    #     limit=10,
    #     with_payload=True,
    # )
    prefetch = [
        models.Prefetch(
            query=dense_query_vector,
            using="all-MiniLM-L6-v2",
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_query_vector.as_object()),
            using="bm25",
            limit=20,
        ),
        models.Prefetch(
            query=late_query_vector,
            using="colbertv2.0",
            limit=20,
        ),
    ]
    results = vector_db.client.query_points(
            "rag2",
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            with_payload=True,
            limit=5,
        )
    # vectorstore = QdrantVectorStore.from_existing_collection(embedding="all-MiniLM-L6-v2", collection_name="rag2", url = Config.QDRANT_URL)
    
    # retriever = vectorstore.as_retriever()
    # docs = retriever.invoke(
    #     "explain Polyline Sequence Representation in SMERF?"
    #     )
    dict = {
        str(point.id): point.score
        for point in results.points
    }
    point_id = [point.id for point in results.points]
    points = vector_db.client.retrieve(collection_name="rag2", ids=point_id)
    for point in points:
        print(f"Point ID: {point.id}, Score: {dict[str(point.id)]}")
        # print(point.payload["metadata"]["dl_meta"]["headings"])
        print(point.payload["text"])