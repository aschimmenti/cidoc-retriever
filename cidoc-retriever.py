import faiss
import voyageai
import numpy as np
from typing import List, Dict, Optional
import json
import os
import argparse
import sys
from openai import OpenAI

class CIDOCPropertyRAG:
    def __init__(self, voyage_api_key: str, openai_client):
        """
        Initialize CIDOC-CRM property search system with RAG capabilities.
        
        Args:
            voyage_api_key: Your Voyage AI API key
            openai_client: OpenAI client for generation
        """
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.openai_client = openai_client
        self.properties = {}
        self.classes = {}
        self.property_chunks = []
        self.class_chunks = []
        self.property_index = None
        self.class_index = None
        
    def load_cidoc_data(self, cidoc_json_path: str):
        """Load CIDOC-CRM data from JSON file."""
        try:
            with open(cidoc_json_path, 'r', encoding='utf-8') as f:
                cidoc_data = json.load(f)
            
            self.properties = cidoc_data.get('properties', {})
            self.classes = cidoc_data.get('classes', {})
            
            print(f"Loaded {len(self.properties)} properties and {len(self.classes)} classes")
            return True
            
        except FileNotFoundError:
            print(f"CIDOC data file not found: {cidoc_json_path}")
            return False
        except Exception as e:
            print(f"Error loading CIDOC data: {e}")
            return False
    
    def create_property_search_chunks(self):
        """
        Create searchable chunks from CIDOC properties for RAG.
        Each property becomes a comprehensive chunk with semantic information.
        """
        self.property_chunks = []
        
        for prop_id, prop_data in self.properties.items():
            # Create comprehensive search text for each property
            search_text_parts = [
                f"Property {prop_id}: {prop_data.get('name', '')}",
                f"Domain: {prop_data.get('domain', '')}",
                f"Range: {prop_data.get('range', '')}",
            ]
            
            # Add scope note if available
            scope_note = prop_data.get('scope_note', '')
            if scope_note:
                search_text_parts.append(f"Description: {scope_note}")
            
            # Add examples if available
            examples = prop_data.get('examples', [])
            if examples:
                search_text_parts.append(f"Examples: {' | '.join(examples[:3])}")
            
            # Add semantic keywords based on property meaning
            keywords = self.extract_semantic_keywords(prop_data)
            if keywords:
                search_text_parts.append(f"Keywords: {', '.join(keywords)}")
            
            chunk = {
                'property_id': prop_id,
                'property_name': prop_data.get('name', ''),
                'domain': prop_data.get('domain', ''),
                'range': prop_data.get('range', ''),
                'scope_note': scope_note,
                'examples': examples,
                'search_text': ' '.join(search_text_parts),
                'uri_forward': prop_data.get('uri_forward', ''),
                'uri_inverse': prop_data.get('uri_inverse', ''),
                'quantification': prop_data.get('quantification', ''),
                'keywords': keywords
            }
            
            self.property_chunks.append(chunk)
        
        print(f"Created {len(self.property_chunks)} property search chunks")
    
    def create_class_search_chunks(self):
        """
        Create searchable chunks from CIDOC classes for RAG.
        Each class becomes a comprehensive chunk with semantic information.
        """
        self.class_chunks = []
        
        for class_id, class_data in self.classes.items():
            # Create comprehensive search text for each class
            search_text_parts = [
                f"Class {class_id}: {class_data.get('name', '')}",
            ]
            
            # Add scope note if available
            scope_note = class_data.get('scope_note', '')
            if scope_note:
                search_text_parts.append(f"Description: {scope_note}")
            
            # Add examples if available
            examples = class_data.get('examples', [])
            if examples:
                search_text_parts.append(f"Examples: {' | '.join(examples[:3])}")
            
            # Add subclass/superclass information
            subclass_of = class_data.get('subclass_of', [])
            if subclass_of:
                search_text_parts.append(f"Subclass of: {', '.join(subclass_of[:3])}")
            
            superclass_of = class_data.get('superclass_of', [])
            if superclass_of:
                search_text_parts.append(f"Superclass of: {', '.join(superclass_of[:3])}")
            
            # Add semantic keywords based on class meaning
            keywords = self.extract_class_semantic_keywords(class_data)
            if keywords:
                search_text_parts.append(f"Keywords: {', '.join(keywords)}")
            
            chunk = {
                'class_id': class_id,
                'class_name': class_data.get('name', ''),
                'scope_note': scope_note,
                'examples': examples,
                'subclass_of': subclass_of,
                'superclass_of': superclass_of,
                'search_text': ' '.join(search_text_parts),
                'uri': class_data.get('uri', ''),
                'keywords': keywords,
                'type': 'class'
            }
            
            self.class_chunks.append(chunk)
        
        print(f"Created {len(self.class_chunks)} class search chunks")
    
    def extract_semantic_keywords(self, prop_data: Dict) -> List[str]:
        """Extract semantic keywords from property data for better matching."""
        keywords = set()
        
        prop_name = prop_data.get('name', '').lower()
        scope_note = prop_data.get('scope_note', '').lower()
        domain = prop_data.get('domain', '').lower()
        range_class = prop_data.get('range', '').lower()
        
        # Define semantic mappings for common concepts
        keyword_mappings = {
            # Identification and naming
            'identified': ['name', 'identifier', 'label', 'title', 'called', 'naming'],
            'type': ['classification', 'category', 'kind', 'class', 'classified'],
            
            # Creation and production
            'created': ['made', 'produced', 'authored', 'wrote', 'composed', 'formed'],
            'carried out': ['performed', 'executed', 'did', 'conducted', 'done by'],
            'produced': ['created', 'manufactured', 'generated', 'made'],
            
            # Spatial relations
            'place': ['location', 'where', 'at', 'in', 'geographic', 'spatial'],
            'took place': ['happened', 'occurred', 'located', 'venue'],
            'residence': ['lived', 'inhabited', 'dwelling', 'home', 'born', 'birthplace'],
            'moved': ['transported', 'relocated', 'transferred location'],
            
            # Temporal relations
            'time': ['when', 'date', 'period', 'temporal', 'duration', 'timing'],
            'birth': ['born', 'birthplace', 'birth location', 'natal'],
            'death': ['died', 'death location', 'mortality'],
            
            # Participation and agency
            'participant': ['involved', 'actor', 'agent', 'person', 'who', 'participated'],
            'actor': ['person', 'agent', 'participant', 'performer'],
            
            # Relationships and references
            'refers to': ['about', 'concerning', 'mentions', 'describes', 'related to'],
            'represents': ['depicts', 'shows', 'portrays', 'symbolizes'],
            'depicts': ['represents', 'shows', 'portrays', 'illustrates'],
            
            # Physical relations
            'custody': ['kept', 'held', 'maintained', 'stored', 'possession'],
            'ownership': ['owned', 'possessed', 'belonged', 'property', 'title'],
            'composed of': ['contains', 'includes', 'made up of', 'parts'],
            
            # Language and communication
            'language': ['linguistic', 'written in', 'spoken', 'text', 'communication'],
            
            # Measurement and dimensions
            'dimension': ['size', 'measurement', 'quantity', 'amount', 'extent'],
            'measured': ['quantified', 'assessed', 'evaluated', 'calculated'],
        }
        
        # Add keywords based on property name and scope
        text_to_analyze = f"{prop_name} {scope_note}"
        
        for concept, synonyms in keyword_mappings.items():
            if concept in text_to_analyze or any(syn in text_to_analyze for syn in synonyms):
                keywords.update(synonyms)
                keywords.add(concept)
        
        # Add domain/range as keywords (simplified)
        if 'person' in domain or 'actor' in domain:
            keywords.update(['person', 'individual', 'people', 'human'])
        if 'place' in range_class or 'location' in range_class:
            keywords.update(['place', 'location', 'where', 'geography'])
        if 'time' in range_class or 'temporal' in range_class:
            keywords.update(['time', 'when', 'temporal', 'date'])
        
        # Special mappings for birth/death relations
        if 'birth' in text_to_analyze:
            keywords.update(['born', 'birthplace', 'birth location', 'natal', 'origin'])
        if 'death' in text_to_analyze:
            keywords.update(['died', 'death location', 'mortality', 'deceased'])
        
        return list(keywords)
    
    def extract_class_semantic_keywords(self, class_data: Dict) -> List[str]:
        """Extract semantic keywords from class data for better matching."""
        keywords = set()
        
        class_name = class_data.get('name', '').lower()
        scope_note = class_data.get('scope_note', '').lower()
        
        # Define semantic mappings for common class concepts
        keyword_mappings = {
            # Entity types
            'person': ['human', 'individual', 'people', 'actor', 'agent', 'biography'],
            'object': ['thing', 'item', 'artifact', 'material', 'physical'],
            'event': ['activity', 'happening', 'occurrence', 'action', 'process'],
            'place': ['location', 'site', 'geography', 'spatial', 'where'],
            'time': ['temporal', 'period', 'date', 'when', 'duration'],
            'document': ['text', 'writing', 'record', 'information', 'linguistic'],
            'type': ['classification', 'category', 'concept', 'taxonomy'],
            
            # Cultural heritage specific
            'artwork': ['art', 'creative', 'cultural', 'aesthetic'],
            'creation': ['production', 'making', 'authorship', 'genesis'],
            'collection': ['museum', 'archive', 'repository', 'curation'],
            'measurement': ['dimension', 'quantity', 'size', 'metric'],
            'identification': ['name', 'label', 'identifier', 'title'],
            
            # Conceptual vs Physical
            'conceptual': ['abstract', 'idea', 'mental', 'symbolic'],
            'physical': ['material', 'tangible', 'concrete', 'spatial'],
            
            # Biological
            'biological': ['living', 'organic', 'natural', 'life'],
            
            # Human-made
            'human-made': ['artificial', 'manufactured', 'created', 'produced'],
        }
        
        # Add keywords based on class name and scope
        text_to_analyze = f"{class_name} {scope_note}"
        
        for concept, synonyms in keyword_mappings.items():
            if concept in text_to_analyze or any(syn in text_to_analyze for syn in synonyms):
                keywords.update(synonyms)
                keywords.add(concept)
        
        # Special handling for common class patterns
        if 'entity' in class_name:
            keywords.update(['entity', 'thing', 'object', 'item'])
        if 'temporal' in text_to_analyze:
            keywords.update(['time', 'temporal', 'when', 'period'])
        if 'spatial' in text_to_analyze:
            keywords.update(['space', 'location', 'place', 'where'])
        if 'actor' in text_to_analyze:
            keywords.update(['person', 'agent', 'actor', 'who'])
        
        return list(keywords)
    
    def create_property_index(self):
        """Create FAISS index for property search."""
        if not self.property_chunks:
            print("No property chunks available. Call create_property_search_chunks() first.")
            return
        
        print("Creating embeddings for property search...")
        
        # Create embeddings for all property chunks
        search_texts = [chunk['search_text'] for chunk in self.property_chunks]
        
        # Use Voyage embeddings for semantic search
        embeddings_obj = self.voyage_client.embed(
            texts=search_texts,
            model="voyage-3-large",
            input_type="document"
        )
        
        embeddings = np.array(embeddings_obj.embeddings).astype('float32')
        
        # Create FAISS index
        dim = embeddings.shape[1]
        self.property_index = faiss.IndexFlatIP(dim)  # Inner product for normalized embeddings
        self.property_index.add(embeddings)
        
        print(f"Created FAISS index with {len(embeddings)} property embeddings")
    
    def create_class_index(self):
        """Create FAISS index for class search."""
        if not self.class_chunks:
            print("No class chunks available. Call create_class_search_chunks() first.")
            return
        
        print("Creating embeddings for class search...")
        
        # Create embeddings for all class chunks
        search_texts = [chunk['search_text'] for chunk in self.class_chunks]
        
        # Use Voyage embeddings for semantic search
        embeddings_obj = self.voyage_client.embed(
            texts=search_texts,
            model="voyage-3-large",
            input_type="document"
        )
        
        embeddings = np.array(embeddings_obj.embeddings).astype('float32')
        
        # Create FAISS index
        dim = embeddings.shape[1]
        self.class_index = faiss.IndexFlatIP(dim)  # Inner product for normalized embeddings
        self.class_index.add(embeddings)
        
        print(f"Created FAISS index with {len(embeddings)} class embeddings")
    
    def search_properties(self, query: str, k: int = 5, use_reranking: bool = True) -> List[Dict]:
        """
        Search for relevant CIDOC properties based on semantic query.
        
        Args:
            query: Natural language query like "property to connect person with birthplace"
            k: Number of properties to return
            use_reranking: Whether to use Voyage reranker for better results
        """
        if not self.property_index:
            print("Property index not created. Call create_property_index() first.")
            return []
        
        # Create query embedding
        query_embedding = self.voyage_client.embed(
            texts=[query],
            model="voyage-3-large", 
            input_type="query"
        ).embeddings[0]
        
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS
        if use_reranking:
            # Get more candidates for reranking
            distances, indices = self.property_index.search(query_embedding, k * 3)
            candidates = [self.property_chunks[i] for i in indices[0]]
            
            # Use Voyage reranker
            rerank_results = self.voyage_client.rerank(
                query=query,
                documents=[c['search_text'] for c in candidates],
                model="rerank-2",
                top_k=k
            )
            
            # Get reranked results
            results = []
            for result in rerank_results.results:
                chunk = candidates[result.index]
                chunk['relevance_score'] = result.relevance_score
                results.append(chunk)
            
            return results
        else:
            # Direct FAISS search
            distances, indices = self.property_index.search(query_embedding, k)
            results = []
            for i, idx in enumerate(indices[0]):
                chunk = self.property_chunks[idx].copy()
                chunk['relevance_score'] = float(distances[0][i])
                results.append(chunk)
            
            return results
    
    def search_classes(self, query: str, k: int = 5, use_reranking: bool = True) -> List[Dict]:
        """
        Search for relevant CIDOC classes based on semantic query.
        
        Args:
            query: Natural language query like "class for representing a person"
            k: Number of classes to return
            use_reranking: Whether to use Voyage reranker for better results
        """
        if not self.class_index:
            print("Class index not created. Call create_class_index() first.")
            return []
        
        # Create query embedding
        query_embedding = self.voyage_client.embed(
            texts=[query],
            model="voyage-3-large", 
            input_type="query"
        ).embeddings[0]
        
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS
        if use_reranking:
            # Get more candidates for reranking
            distances, indices = self.class_index.search(query_embedding, k * 3)
            candidates = [self.class_chunks[i] for i in indices[0]]
            
            # Use Voyage reranker
            rerank_results = self.voyage_client.rerank(
                query=query,
                documents=[c['search_text'] for c in candidates],
                model="rerank-2",
                top_k=k
            )
            
            # Get reranked results
            results = []
            for result in rerank_results.results:
                chunk = candidates[result.index]
                chunk['relevance_score'] = result.relevance_score
                results.append(chunk)
            
            return results
        else:
            # Direct FAISS search
            distances, indices = self.class_index.search(query_embedding, k)
            results = []
            for i, idx in enumerate(indices[0]):
                chunk = self.class_chunks[idx].copy()
                chunk['relevance_score'] = float(distances[0][i])
                results.append(chunk)
            
            return results
    
    def search_combined(self, query: str, k_properties: int = 3, k_classes: int = 3, use_reranking: bool = True) -> Dict[str, List[Dict]]:
        """
        Search for both relevant CIDOC properties and classes based on semantic query.
        
        Args:
            query: Natural language query
            k_properties: Number of properties to return
            k_classes: Number of classes to return
            use_reranking: Whether to use Voyage reranker for better results
        
        Returns:
            Dictionary with 'properties' and 'classes' keys containing respective results
        """
        results = {
            'properties': [],
            'classes': []
        }
        
        # Search properties
        if self.property_index:
            results['properties'] = self.search_properties(query, k=k_properties, use_reranking=use_reranking)
        
        # Search classes
        if self.class_index:
            results['classes'] = self.search_classes(query, k=k_classes, use_reranking=use_reranking)
        
        return results
    
    def explain_combined_choice(self, query: str, combined_results: Dict[str, List[Dict]]) -> str:
        """
        Use LLM to explain why certain properties and classes are relevant for the query.
        """
        properties = combined_results.get('properties', [])
        classes = combined_results.get('classes', [])
        
        if not properties and not classes:
            return "No relevant CIDOC-CRM properties or classes found for your query."
        
        # Format results for the explanation
        explanation_parts = []
        
        if properties:
            properties_info = []
            for i, result in enumerate(properties[:3], 1):  # Top 3 results
                prop_info = f"""
Property {i}: {result['property_id']} - {result['property_name']}
Domain: {result['domain']}
Range: {result['range']}
Description: {result['scope_note'][:200]}{'...' if len(result['scope_note']) > 200 else ''}
Relevance Score: {result['relevance_score']:.3f}
"""
                properties_info.append(prop_info)
            explanation_parts.append(f"Top relevant CIDOC-CRM properties:\n{''.join(properties_info)}")
        
        if classes:
            classes_info = []
            for i, result in enumerate(classes[:3], 1):  # Top 3 results
                class_info = f"""
Class {i}: {result['class_id']} - {result['class_name']}
Description: {result['scope_note'][:200]}{'...' if len(result['scope_note']) > 200 else ''}
Subclass of: {', '.join(result['subclass_of'][:2]) if result['subclass_of'] else 'None'}
Relevance Score: {result['relevance_score']:.3f}
"""
                classes_info.append(class_info)
            explanation_parts.append(f"Top relevant CIDOC-CRM classes:\n{''.join(classes_info)}")
        
        prompt = f"""You are an expert in CIDOC-CRM ontology helping users find the right properties and classes for modeling cultural heritage data.

User Query: "{query}"

{chr(10).join(explanation_parts)}

Please explain:
1. Which properties and/or classes best answer the user's query and why
2. How each property should be used (domain → property → range) and how classes fit into the model
3. How properties and classes work together in CIDOC-CRM modeling
4. Any important considerations or alternatives
5. Provide a concrete example showing both classes and properties in use

Keep your explanation clear, practical, and focused on helping the user understand how to model their data correctly."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a CIDOC-CRM expert helping users understand how to use the ontology for cultural heritage data modeling."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
    def ask_cidoc_question(self, question: str, k_properties: int = 3, k_classes: int = 3) -> str:
        """
        Answer a natural language question about CIDOC-CRM properties and classes.
        
        Args:
            question: Natural language question about properties and classes
            k_properties: Number of properties to search
            k_classes: Number of classes to search
        """
        # Search for relevant properties and classes
        results = self.search_combined(question, k_properties=k_properties, k_classes=k_classes)
        
        if not results['properties'] and not results['classes']:
            return "I couldn't find any relevant CIDOC-CRM properties or classes for your query. Please try rephrasing your question."
        
        # Get detailed explanation
        explanation = self.explain_combined_choice(question, results)
        
        return explanation
    
    def ask_property_question(self, question: str, k: int = 5) -> str:
        """
        Answer a natural language question about CIDOC-CRM properties.
        
        Args:
            question: Natural language question about properties
            k: Number of properties to search
        """
        # Search for relevant properties
        results = self.search_properties(question, k=k)
        
        if not results:
            return "I couldn't find any relevant CIDOC-CRM properties for your query. Please try rephrasing your question."
        
        # Get detailed explanation using combined method for consistency
        combined_results = {'properties': results, 'classes': []}
        explanation = self.explain_combined_choice(question, combined_results)
        
        return explanation
    
    def save_indices(self, 
                    property_index_path: str = "cidoc_property_index.faiss", 
                    property_metadata_path: str = "cidoc_property_chunks.json",
                    class_index_path: str = "cidoc_class_index.faiss",
                    class_metadata_path: str = "cidoc_class_chunks.json"):
        """Save both property and class indices and chunks for later use."""
        saved_any = False
        
        if self.property_index:
            faiss.write_index(self.property_index, property_index_path)
            with open(property_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.property_chunks, f, ensure_ascii=False, indent=2)
            print(f"Saved property index to {property_index_path} and chunks to {property_metadata_path}")
            saved_any = True
        
        if self.class_index:
            faiss.write_index(self.class_index, class_index_path)
            with open(class_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.class_chunks, f, ensure_ascii=False, indent=2)
            print(f"Saved class index to {class_index_path} and chunks to {class_metadata_path}")
            saved_any = True
        
        if not saved_any:
            print("No indices to save")
    
    def load_indices(self, 
                    property_index_path: str = "cidoc_property_index.faiss",
                    property_metadata_path: str = "cidoc_property_chunks.json",
                    class_index_path: str = "cidoc_class_index.faiss", 
                    class_metadata_path: str = "cidoc_class_chunks.json"):
        """Load previously saved property and class indices."""
        print("Looking for existing indices in local folder...")
        loaded_any = False
        
        # Try to load property index
        try:
            if not os.path.exists(property_index_path) or not os.path.exists(property_metadata_path):
                print("Property index files not found in local folder")
            else:
                self.property_index = faiss.read_index(property_index_path)
                with open(property_metadata_path, 'r', encoding='utf-8') as f:
                    self.property_chunks = json.load(f)
                print(f"✅ Loaded property index with {len(self.property_chunks)} properties")
                loaded_any = True
        except Exception as e:
            print(f"Property index could not be loaded: {str(e).split(':')[-1].strip()}")
        
        # Try to load class index
        try:
            if not os.path.exists(class_index_path) or not os.path.exists(class_metadata_path):
                print("Class index files not found in local folder")
            else:
                self.class_index = faiss.read_index(class_index_path)
                with open(class_metadata_path, 'r', encoding='utf-8') as f:
                    self.class_chunks = json.load(f)
                print(f"✅ Loaded class index with {len(self.class_chunks)} classes")
                loaded_any = True
        except Exception as e:
            print(f"Class index could not be loaded: {str(e).split(':')[-1].strip()}")
        
        if not loaded_any:
            print("No existing indices found. Will create new FAISS indices...")
        
        return loaded_any
    
    # Backward compatibility methods
    def save_property_index(self, index_path: str = "cidoc_property_index.faiss", 
                           metadata_path: str = "cidoc_property_chunks.json"):
        """Save the property index and chunks for later use."""
        if self.property_index:
            faiss.write_index(self.property_index, index_path)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.property_chunks, f, ensure_ascii=False, indent=2)
            
            print(f"Saved property index to {index_path} and chunks to {metadata_path}")
    
    def load_property_index(self, index_path: str = "cidoc_property_index.faiss",
                           metadata_path: str = "cidoc_property_chunks.json"):
        """Load a previously saved property index."""
        try:
            self.property_index = faiss.read_index(index_path)
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.property_chunks = json.load(f)
            
            print(f"Loaded property index with {len(self.property_chunks)} properties")
            return True
        except Exception as e:
            print(f"Error loading property index: {e}")
            return False

def main():
    """
    Main function to demonstrate CIDOC property search with command line input.
    """
    parser = argparse.ArgumentParser(description='CIDOC-CRM Property Search System')
    parser.add_argument('query', nargs='?', help='Query to search for CIDOC-CRM properties')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--k', type=int, default=5, help='Number of properties to return (default: 5)')
    
    args = parser.parse_args()
    
    # Check for required environment variables
    voyage_api_key = os.getenv('VOYAGE_API_KEY')
    if not voyage_api_key:
        print("❌ Error: VOYAGE_API_KEY environment variable not set.")
        print("Please set your Voyage AI API key: export VOYAGE_API_KEY='your_key_here'")
        sys.exit(1)
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your_key_here'")
        sys.exit(1)
    
    # Initialize clients
    openai_client = OpenAI()  # Uses OPENAI_API_KEY env variable
    
    # Create property search system
    cidoc_rag = CIDOCPropertyRAG(
        voyage_api_key=voyage_api_key,
        openai_client=openai_client
    )
    
    # Load CIDOC data
    if not cidoc_rag.load_cidoc_data("cidoc_crm_parsed.json"):
        print("Failed to load CIDOC data. Make sure to run the parser first.")
        return
    
    # Try to load existing indices first
    if not cidoc_rag.load_indices():
        print("Creating new search indices...")
        cidoc_rag.create_property_search_chunks()
        cidoc_rag.create_property_index()
        cidoc_rag.create_class_search_chunks()
        cidoc_rag.create_class_index()
        cidoc_rag.save_indices()
    
    print(f"\n{'='*80}")
    print("CIDOC-CRM ONTOLOGY RAG SYSTEM")
    print(f"{'='*80}\n")
    
    if args.interactive:
        # Interactive mode
        print("Interactive mode. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                    
                print(f"\nQuery: {query}")
                print("-" * 60)
                
                answer = cidoc_rag.ask_cidoc_question(query, k_properties=args.k//2+1, k_classes=args.k//2+1)
                print(f"{answer}\n")
                print("=" * 80)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    elif args.query:
        # Single query mode
        print(f"Query: {args.query}")
        print("-" * 60)
        
        answer = cidoc_rag.ask_cidoc_question(args.query, k_properties=args.k//2+1, k_classes=args.k//2+1)
        print(f"{answer}\n")
        print("=" * 80)
    else:
        # No query provided, show usage
        parser.print_help()
        print("\nExample usage:")
        print("  python cidoc-retriever.py 'property to connect person with birthplace'")
        print("  python cidoc-retriever.py --interactive")
        print("  python cidoc-retriever.py 'creation properties' --k 3")

if __name__ == "__main__":
    main()