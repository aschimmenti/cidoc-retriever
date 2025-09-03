# CIDOC-CRM Ontology RAG System

A powerful Retrieval-Augmented Generation (RAG) system for semantic search and intelligent querying of the CIDOC-CRM ontology. This system helps cultural heritage professionals, researchers, and developers find the right CIDOC-CRM properties and classes for modeling their data. Currently relying on OpenAI and Voyage AI (soon to be replaced by HuggingFace and Ollama). 

## 🚀 Features

- **Semantic Search**: Natural language queries for CIDOC-CRM properties and classes
- **Combined Search**: Searches both properties and classes simultaneously
- **Interactive Mode**: Real-time query interface for exploration
- **FAISS Indexing**: Fast vector similarity search with 160 properties and 81 classes
- **Keyword Enhancement**: Semantic keyword mapping for improved search accuracy

## 📁 Repository Structure

```
cidoc-RAG/
├── cidoc_parser.py              # Parser for CIDOC-CRM HTML specification
├── cidoc-retriever.py           # Main RAG system with CLI interface
├── cidoc_crm_parsed.json        # Parsed CIDOC-CRM data (properties & classes)
├── README.md                    # This file
├── cidoc_property_index.faiss   # FAISS index for properties (auto-generated)
├── cidoc_class_index.faiss      # FAISS index for classes (auto-generated)
├── cidoc_property_chunks.json   # Property search chunks (auto-generated)
└── cidoc_class_chunks.json      # Class search chunks (auto-generated)
```

## 🛠️ Installation

### Prerequisites

1. **Python 3.8+** with pip
2. **API Keys** for Voyage AI and OpenAI

### Install Dependencies

```bash
pip install faiss-cpu voyageai openai numpy beautifulsoup4 requests
```

### Set Up API Keys

```bash
export VOYAGE_API_KEY="your_voyage_ai_key_here"
export OPENAI_API_KEY="your_openai_key_here"
```

**🔒 Security Note**: Never hardcode API keys in source code. Always use environment variables.

## 🎯 Usage

### Command Line Interface

#### Single Query Mode
```bash
python cidoc-retriever.py "how to model a person creating an artwork"
```

#### Interactive Mode
```bash
python cidoc-retriever.py --interactive
```

#### Specify Number of Results
```bash
python cidoc-retriever.py "temporal properties" --k 6
```

### Example Queries

- **Modeling Questions**: "how to represent an artwork entity and its creation"
- **Property Search**: "property to connect person with birthplace"
- **Class Search**: "class for representing a museum object"
- **Relationship Queries**: "properties for temporal relationships"
- **Domain-Specific**: "modeling the provenance of cultural objects"

## 🧠 How It Works

### 1. Data Processing
- **Parser**: Extracts CIDOC-CRM v7.1.3 data from official HTML specification
- **Chunking**: Creates semantic search chunks for 160 properties and 81 classes
- **Keywords**: Adds domain-specific synonyms and related terms

### 2. Semantic Search
- **Embeddings**: Uses Voyage AI's `voyage-3-large` model for vector embeddings
- **Indexing**: FAISS vector database for fast similarity search
- **Reranking**: Voyage AI's `rerank-2` model for improved relevance

### 3. AI Generation
- **Context**: Combines search results with domain expertise
- **Explanation**: LLM provides detailed, practical explanations
- **Examples**: Concrete usage examples with proper CIDOC-CRM modeling

## 📊 System Capabilities

| Feature | Count | Description |
|---------|-------|-------------|
| Properties | 160 | CIDOC-CRM properties with full metadata |
| Classes | 81 | CIDOC-CRM classes with hierarchical relationships |
| Semantic Keywords | 1000+ | Domain-specific synonyms and related terms |
| Search Modes | 3 | Combined, property-only, class-only search |

## 🔧 Advanced Configuration

### Custom Index Paths
```python
# Load with custom paths
cidoc_rag.load_indices(
    property_index_path="custom_prop.faiss",
    class_index_path="custom_class.faiss"
)
```

### Search Parameters
```python
# Fine-tune search results
results = cidoc_rag.search_combined(
    query="your query",
    k_properties=5,
    k_classes=3,
    use_reranking=True
)
```

## 📈 Performance

- **Index Creation**: ~2-3 minutes (first run only)
- **Query Response**: <2 seconds for most queries
- **Memory Usage**: ~500MB with loaded indices
- **Accuracy**: Enhanced by semantic keywords and reranking

## 🎓 Example Output

```
Query: how to model a person creating an artwork
------------------------------------------------------------

Top relevant CIDOC-CRM properties:
Property 1: P14 - carried out by (performed)
Domain: E7 Activity
Range: E39 Actor
Description: This property describes the active participation of an E39 Actor in an E7 Activity...
Relevance Score: 0.892

Top relevant CIDOC-CRM classes:
Class 1: E12 - Production
Description: This class comprises activities that are designed to, and succeed in, creating...
Subclass of: E11 Modification, E63 Beginning of Existence
Relevance Score: 0.856

[Detailed AI explanation follows...]
```

## 🗺️ Future Roadmap

- [ ] **Multi-Model Support**: Make more models available for embeddings and LLMs (currently relying on Voyage and OpenAI API)
- [ ] **Answer Evaluation**: Evaluate the relevance of the answers with automated scoring and feedback mechanisms
- [ ] **Examples Index**: Add a 'examples' index of KG examples in a separate index for more efficient contextualization
- [ ] **MCP Server Integration**: Develop the tool as a MCP server to be interactive with an agent for CIDOC-CRM KG generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or feature requests, please open an issue on GitHub or contact the maintainers.

---

**Built for the cultural heritage community** 🏛️
