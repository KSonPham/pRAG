import uuid
import hashlib
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from typing import List, Dict, Optional

class PostgresDocStore:
    def __init__(self, dbname: str, user: str, password: str, host: str, port: int = 5432):
        self.conn_params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }
        self._initialize_schema()

    def _get_connection(self):
        try:
            conn = psycopg2.connect(**self.conn_params)
            return conn
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def _initialize_schema(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_title ON document_chunks (title);
        CREATE INDEX IF NOT EXISTS idx_metadata ON document_chunks USING GIN (metadata);
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
            conn.commit()

    def add_documents(self, documents: List[Dict]):
        """
        Store document chunks in PostgreSQL.
        Each document should be a dictionary with:
        - 'id': UUID (string or UUID object)
        - 'title': string
        - 'content': string
        - 'metadata': dict (optional)
        """
        query = """
            INSERT INTO document_chunks (id, title, content, metadata)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata;
        """
        
        data = []
        for doc in documents:
            if not all(key in doc for key in ('id', 'title', 'content')):
                raise ValueError("Document missing required fields (id, title, content)")
                
            doc_id = uuid.UUID(doc['id']) if isinstance(doc['id'], str) else doc['id']
            data.append((
                doc_id,
                doc['title'],
                doc['content'],
                doc.get('metadata', None)
            ))
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                execute_values(
                    cursor,
                    query,
                    data,
                    template="(%s, %s, %s, %s::jsonb)"
                )
            conn.commit()

    def get_document(self, doc_id: uuid.UUID) -> Optional[Dict]:
        """Retrieve a document chunk by its UUID"""
        query = """
            SELECT id, title, content, metadata
            FROM document_chunks
            WHERE id = %s
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (str(doc_id),))
                result = cursor.fetchone()
        
        if result:
            return {
                'id': result[0],
                'title': result[1],
                'content': result[2],
                'metadata': result[3]
            }
        return None

    def search_by_title(self, title: str, limit: int = 10) -> List[Dict]:
        """Search document chunks by title"""
        query = """
            SELECT id, title, content, metadata
            FROM document_chunks
            WHERE title ILIKE %s
            LIMIT %s
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (f'%{title}%', limit))
                results = cursor.fetchall()
        
        return [{
            'id': row[0],
            'title': row[1],
            'content': row[2],
            'metadata': row[3]
        } for row in results]

    def delete_document(self, doc_id: uuid.UUID) -> bool:
        """Delete a document chunk by its UUID"""
        query = """
            DELETE FROM document_chunks
            WHERE id = %s
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (str(doc_id),))
                deleted = cursor.rowcount > 0
            conn.commit()
        
        return deleted

# Example usage with langchain text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

def store_document(store: PostgresDocStore, title: str, content: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100
    )
    
    chunks = text_splitter.create_documents([content])
    
    documents_to_store = []
    for chunk in chunks:
        chunk_id = hashlib.sha256(chunk.page_content.encode()).hexdigest()
        documents_to_store.append({
            'id': chunk_id,
            'title': title,
            'content': chunk.page_content,
            'metadata': {
                'source': title,
                'chunk_number': len(documents_to_store) + 1
            }
        })
    
    store.add_documents(documents_to_store)
    
# Initialize store
doc_store = PostgresDocStore(
    dbname="admin.collection",
    user="postgres",
    password="postgres",
    host="localhost"
)
# Read example.txt file
def read_example_file():
    try:
        with open("example.txt", "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print("Error: example.txt file not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

example_content = read_example_file()
if example_content:
    store_document(doc_store, "example.txt", example_content)

# Store a document

# Retrieve a document
# Get all document IDs directly from DB for better performance
docs = doc_store.client.query(
    "SELECT id FROM documents",
    fetch_all=True
)
doc_uuids = [doc[0] for doc in docs]

# Search documents
results = doc_store.search_by_title("example.txt")