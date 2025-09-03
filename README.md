# cidoc-retriever
A simple RAG + reranker retriever tool for CIDOC-CRM properties. 
It is composed of two scripts: 
- cidoc-parser.py
- cidoc-retriever.py
Cidoc-parser.py parses the Classes & Properties Declarations of CIDOC-CRM (version 7.1.3) and creates a JSON file.
The cidoc-retriver.py returns classes and properties based on a question.
As of September 2025 the tool is still in development and it has not been tested. 
