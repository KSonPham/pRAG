# pRAG

pRAG is a project designed to facilitate question-answering within scientific documentation, making it an invaluable tool for students working on essays, research papers, or theses. By leveraging advanced retrieval and generation techniques, pRAG enables users to efficiently extract relevant information from extensive academic sources, streamlining the research and writing process.

## Features

- **Database Management**: Scripts and configurations for managing the database ([`db.py`](db.py), [`start_db.sh`](start_db.sh)).
- **Application Logic**: Core application logic implemented in [`app.py`](app.py).
- **Configuration**: Centralized configuration management in [`config.py`](config.py).
- **Docker Support**: Dockerized setup with [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml).
- **Chainlit Integration**: Chainlit configuration in [`.chainlit/`](.chainlit/).

## Things To Try:
- **Metadata Filtering**: Exclude queries based on document names to refine search results.  
- **Advanced Retrieval**: Implement a "small-to-big" approach—generate smaller chunks by summarizing or semantically segmenting larger sections. Use these smaller chunks for retrieval and leverage the full chunks for synthesis.  
- **Multi-Document Agents**: Enable fact-based question answering and summarization across multiple documents. Develop an agent that selects relevant documents, applies metadata filtering, and then processes the information accordingly.  

- 
<!-- ## Project Structure -->
