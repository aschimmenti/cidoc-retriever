import re
import json
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import requests

def parse_cidoc_html_from_url(url: str) -> Dict:
    """
    Parse CIDOC-CRM directly from the official URL.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return parse_cidoc_html(response.text)
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return {}

def parse_cidoc_html(html_content: str) -> Dict:
    """
    Parse the specific CIDOC-CRM HTML structure with nested tables.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    result = {
        "properties": {},
        "classes": {},
        "metadata": {
            "version": "7.1.3",
            "source": "https://cidoc-crm.org/html/cidoc_crm_v7.1.3.html",
            "parsed_sections": []
        }
    }
    
    # Parse properties - look for spans with class "prop"
    result["properties"] = parse_properties_from_tables(soup)
    result["metadata"]["parsed_sections"].append("properties")
    
    # Parse classes - look for spans with class "cls"
    result["classes"] = parse_classes_from_tables(soup)  
    result["metadata"]["parsed_sections"].append("classes")
    
    return result

def parse_properties_from_tables(soup: BeautifulSoup) -> Dict:
    """
    Parse properties from HTML table structure.
    Properties are identified by <span class="prop" id="P...">
    """
    properties = {}
    
    # Find all property spans
    prop_spans = soup.find_all('span', class_='prop')
    
    for prop_span in prop_spans:
        prop_id = prop_span.get('id', '')
        if not prop_id.startswith('P'):
            continue
            
        # Get the property name from the span text
        prop_text = prop_span.get_text(strip=True)
        
        # Extract property name (everything after the ID)
        prop_name_match = re.match(rf'{re.escape(prop_id)}\s+(.+)', prop_text)
        prop_name = prop_name_match.group(1) if prop_name_match else prop_text
        
        # Find the containing table structure
        prop_table = find_parent_table(prop_span)
        if prop_table:
            property_data = parse_property_from_table(prop_id, prop_name, prop_table)
            if property_data:
                properties[prop_id] = property_data
    
    return properties

def parse_classes_from_tables(soup: BeautifulSoup) -> Dict:
    """
    Parse classes from HTML table structure.
    Classes are identified by <span class="cls" id="E...">
    """
    classes = {}
    
    # Find all class spans
    class_spans = soup.find_all('span', class_='cls')
    
    for class_span in class_spans:
        class_id = class_span.get('id', '')
        if not class_id.startswith('E'):
            continue
            
        # Get the class name from the span text
        class_text = class_span.get_text(strip=True)
        
        # Extract class name (everything after the ID)
        class_name_match = re.match(rf'{re.escape(class_id)}\s+(.+)', class_text)
        class_name = class_name_match.group(1) if class_name_match else class_text
        
        # Remove "(show all properties)" if present
        class_name = re.sub(r'\s*\(show all properties\)\s*$', '', class_name)
        
        # Find the containing table structure
        class_table = find_parent_table(class_span)
        if class_table:
            class_data = parse_class_from_table(class_id, class_name, class_table)
            if class_data:
                classes[class_id] = class_data
    
    return classes

def find_parent_table(element) -> Optional:
    """
    Find the parent table that contains the full definition.
    """
    parent = element.parent
    while parent:
        if parent.name == 'table':
            return parent
        parent = parent.parent
    return None

def parse_property_from_table(prop_id: str, prop_name: str, table) -> Optional[Dict]:
    """
    Parse property data from its table structure.
    Combines original URI extraction with improved content parsing.
    """
    property_data = {
        "id": prop_id,
        "name": prop_name,
        "uri_forward": "",
        "uri_inverse": "",
        "domain": "",
        "domain_uri": "",
        "range": "",
        "range_uri": "",
        "scope_note": "",
        "examples": [],
        "quantification": "",
        "subproperty_of": "",
        "superproperty_of": []
    }
    
    # Find all rows in the table
    rows = table.find_all('tr')
    
    # First pass: use original logic for URIs (which was working)
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) < 2:
            continue
            
        label_cell = cells[0]
        content_cell = cells[1] if len(cells) > 1 else cells[0]
        
        label_text = label_cell.get_text(strip=True)
        
        # Original URI extraction logic that was working
        if 'URI (forward direction):' in label_text:
            uri_link = content_cell.find('a')
            if uri_link:
                property_data["uri_forward"] = uri_link.get('href', '')
            else:
                property_data["uri_forward"] = content_cell.get_text(strip=True)
                
        elif 'URI (inverse direction):' in label_text:
            uri_link = content_cell.find('a')
            if uri_link:
                property_data["uri_inverse"] = uri_link.get('href', '')
            else:
                inverse_text = content_cell.get_text(strip=True)
                property_data["uri_inverse"] = inverse_text if inverse_text != '-' else ""
    
    # Second pass: use improved logic for content fields
    i = 0
    while i < len(rows):
        row = rows[i]
        
        # Look for cardLabel spans that indicate field types
        card_label = row.find('span', class_='cardLabel')
        if not card_label:
            i += 1
            continue
            
        label_text = card_label.get_text(strip=True)
        
        # Parse content fields with improved logic
        if label_text == 'Domain:':
            # Domain info is usually in the next row
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                domain_text = next_row.get_text(strip=True)
                # Clean up domain text - remove duplicate class IDs
                domain_text = clean_class_reference(domain_text)
                property_data["domain"] = domain_text
                
                # Extract domain URI from link
                domain_link = next_row.find('a')
                if domain_link:
                    fragment_uri = domain_link.get('href', '')
                    property_data["domain_uri"] = resolve_class_uri(fragment_uri)
                
        elif label_text == 'Range:':
            # Range info is usually in the next row
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                range_text = next_row.get_text(strip=True)
                # Clean up range text - remove duplicate class IDs
                range_text = clean_class_reference(range_text)
                property_data["range"] = range_text
                
                # Extract range URI from link
                range_link = next_row.find('a')
                if range_link:
                    fragment_uri = range_link.get('href', '')
                    property_data["range_uri"] = resolve_class_uri(fragment_uri)
                
        elif label_text == 'Quantification:':
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                quant_text = next_row.get_text(strip=True)
                property_data["quantification"] = quant_text
                
        elif label_text == 'SubProperty Of:':
            # Collect all subsequent rows until we hit another label
            subprop_chains = []
            subprop_chain_uris = []
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                if next_row.find('span', class_='cardLabel'):
                    break
                subprop_text = next_row.get_text(strip=True)
                if subprop_text and subprop_text != '-':
                    subprop_chains.append(subprop_text)
                    
                    # Extract URIs from each individual property chain pattern
                    subprop_links = next_row.find_all('a')
                    chain_uris = {
                        "entities": [],
                        "properties": []
                    }
                    
                    # Use sets to avoid duplicates within each chain
                    entity_set = set()
                    property_set = set()
                    
                    for link in subprop_links:
                        href = link.get('href', '')
                        if href:
                            if '#E' in href:
                                # Extract entity ID and construct URI
                                entity_id = href.split('#')[-1]
                                if entity_id.startswith('E'):
                                    entity_uri = f"http://www.cidoc-crm.org/cidoc-crm/{entity_id}"
                                    entity_set.add(entity_uri)
                            elif '#P' in href:
                                # Extract property ID and construct URI
                                prop_id = href.split('#')[-1]
                                if prop_id.startswith('P'):
                                    property_uri = f"http://www.cidoc-crm.org/cidoc-crm/{prop_id}"
                                    property_set.add(property_uri)
                    
                    # Convert sets back to lists to maintain order
                    chain_uris["entities"] = list(entity_set)
                    chain_uris["properties"] = list(property_set)
                    
                    if chain_uris["entities"] or chain_uris["properties"]:
                        subprop_chain_uris.append(chain_uris)
                j += 1
            
            # Store the property chains
            if subprop_chains:
                if len(subprop_chains) == 1:
                    property_data["subproperty_of"] = subprop_chains[0]
                else:
                    property_data["subproperty_of"] = subprop_chains
                
                if subprop_chain_uris:
                    property_data["subproperty_of_chain_uris"] = subprop_chain_uris
                    
        elif label_text == 'SuperProperty Of:':
            # Collect all subsequent rows until we hit another label
            superprops = []
            superprop_uris = []
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                if next_row.find('span', class_='cardLabel'):
                    break
                superprop_text = next_row.get_text(strip=True)
                if superprop_text and superprop_text != '-':
                    # Clean up superproperty text - remove duplicate class IDs
                    superprop_text = clean_superproperty_text(superprop_text)
                    superprops.append(superprop_text)
                    # Extract superproperty URIs from links
                    superprop_links = next_row.find_all('a')
                    for link in superprop_links:
                        href = link.get('href', '')
                        if href and '#P' in href:
                            # Extract property ID and construct URI
                            prop_id = href.split('#')[-1]
                            if prop_id.startswith('P') and prop_id not in [uri.split('/')[-1] for uri in superprop_uris]:
                                superprop_uris.append(f"http://www.cidoc-crm.org/cidoc-crm/{prop_id}")
                j += 1
            property_data["superproperty_of"] = superprops
            if superprop_uris:
                property_data["superproperty_of_uris"] = superprop_uris
            
        elif label_text == 'Scope Note:':
            # Collect all text until we hit Examples or another label
            scope_parts = []
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                # Stop if we hit another label
                if next_row.find('span', class_='cardLabel'):
                    break
                # Get text content, handling paragraphs and lists
                scope_text = clean_html_content(next_row.get_text())
                if scope_text:
                    scope_parts.append(scope_text)
                j += 1
            property_data["scope_note"] = ' '.join(scope_parts)
            
        elif label_text == 'Examples:':
            # Look for ul/li elements in subsequent rows
            examples = []
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                if next_row.find('span', class_='cardLabel'):
                    break
                # Look for list items
                li_elements = next_row.find_all('li')
                for li in li_elements:
                    example_text = li.get_text(strip=True)
                    if example_text:
                        examples.append(example_text)
                j += 1
            property_data["examples"] = examples
        
        i += 1
    
    return property_data

def parse_class_from_table(class_id: str, class_name: str, table) -> Optional[Dict]:
    """
    Parse class data from its table structure.
    """
    class_data = {
        "id": class_id,
        "name": class_name,
        "uri": "",
        "subclass_of": [],
        "subclass_of_uris": [],
        "superclass_of": [],
        "superclass_of_uris": [],
        "scope_note": "",
        "examples": []
    }
    
    # Find all rows in the table
    rows = table.find_all('tr')
    
    i = 0
    while i < len(rows):
        row = rows[i]
        
        # Look for cardLabel spans that indicate field types
        card_label = row.find('span', class_='cardLabel')
        if not card_label:
            i += 1
            continue
            
        label_text = card_label.get_text(strip=True)
        
        # Parse based on label type
        if label_text == 'URI:':
            # Look for the URI link in the same row
            uri_link = row.find('a')
            if uri_link:
                class_data["uri"] = uri_link.get('href', '')
                
        elif label_text == 'SubClass Of:':
            # Collect subclasses from subsequent rows (like superclass logic)
            subclasses = []
            subclass_uris = []
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                if next_row.find('span', class_='cardLabel'):
                    break
                subclass_text = next_row.get_text(strip=True)
                if subclass_text and subclass_text != '-':
                    # Clean and split cramped subclass text
                    cleaned_subclasses = parse_cramped_superclass_text(subclass_text)
                    subclasses.extend(cleaned_subclasses)
                    
                    # Extract URIs from links
                    class_links = next_row.find_all('a')
                    for link in class_links:
                        fragment_uri = link.get('href', '')
                        if fragment_uri:
                            resolved_uri = resolve_class_uri(fragment_uri)
                            if resolved_uri not in subclass_uris:
                                subclass_uris.append(resolved_uri)
                j += 1
            
            # Remove duplicates from subclasses list while preserving order
            seen_classes = set()
            unique_subclasses = []
            for sc in subclasses:
                if sc not in seen_classes:
                    unique_subclasses.append(sc)
                    seen_classes.add(sc)
            
            class_data["subclass_of"] = unique_subclasses
            class_data["subclass_of_uris"] = subclass_uris
                        
        elif label_text == 'SuperClass Of:':
            # Collect superclasses from subsequent rows
            superclasses = []
            superclass_uris = []
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                if next_row.find('span', class_='cardLabel'):
                    break
                superclass_text = next_row.get_text(strip=True)
                if superclass_text and superclass_text != '-':
                    # Clean and split cramped superclass text
                    cleaned_superclasses = parse_cramped_superclass_text(superclass_text)
                    superclasses.extend(cleaned_superclasses)
                    
                    # Extract URIs from links
                    class_links = next_row.find_all('a')
                    for link in class_links:
                        fragment_uri = link.get('href', '')
                        if fragment_uri:
                            resolved_uri = resolve_class_uri(fragment_uri)
                            if resolved_uri not in superclass_uris:
                                superclass_uris.append(resolved_uri)
                j += 1
            
            # Remove duplicates from superclasses list while preserving order
            seen_classes = set()
            unique_superclasses = []
            for sc in superclasses:
                if sc not in seen_classes:
                    unique_superclasses.append(sc)
                    seen_classes.add(sc)
            
            class_data["superclass_of"] = unique_superclasses
            class_data["superclass_of_uris"] = superclass_uris
            
        elif label_text == 'Scope Note:':
            # Collect scope note from subsequent rows
            scope_parts = []
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                # Stop if we hit another label
                if next_row.find('span', class_='cardLabel'):
                    break
                # Get text content, handling paragraphs
                for p in next_row.find_all('p'):
                    scope_text = clean_html_content(p.get_text())
                    if scope_text:
                        scope_parts.append(scope_text)
                # Also get any direct text
                if not next_row.find_all('p'):
                    scope_text = clean_html_content(next_row.get_text())
                    if scope_text:
                        scope_parts.append(scope_text)
                j += 1
            class_data["scope_note"] = ' '.join(scope_parts)
            
        elif label_text == 'Examples:':
            # Look for ul/li elements in subsequent rows
            examples = []
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                if next_row.find('span', class_='cardLabel'):
                    break
                # Look for list items
                li_elements = next_row.find_all('li')
                for li in li_elements:
                    example_text = li.get_text(strip=True)
                    if example_text:
                        examples.append(example_text)
                j += 1
            class_data["examples"] = examples
        
        i += 1
    
    return class_data

def extract_content_from_subsequent_rows(start_row, all_rows, stop_at: str = None) -> str:
    """
    Extract content from rows following the start_row until we hit stop_at text.
    """
    content_parts = []
    start_idx = all_rows.index(start_row)
    
    for row in all_rows[start_idx + 1:]:
        row_text = row.get_text(strip=True)
        
        # Stop if we hit the stop condition
        if stop_at and stop_at in row_text:
            break
            
        # Stop if we hit another label
        if row.find('span', class_='cardLabel'):
            break
            
        if row_text:
            content_parts.append(row_text)
    
    return ' '.join(content_parts)

def extract_single_value_from_subsequent_rows(start_row, all_rows) -> str:
    """
    Extract a single value from the next row after start_row.
    """
    start_idx = all_rows.index(start_row)
    
    if start_idx + 1 < len(all_rows):
        next_row = all_rows[start_idx + 1]
        # Look for links or just text
        link = next_row.find('a')
        if link:
            return link.get_text(strip=True)
        else:
            return next_row.get_text(strip=True)
    
    return ""

def extract_list_from_subsequent_rows(start_row, all_rows) -> List[str]:
    """
    Extract a list of items from subsequent rows.
    """
    items = []
    start_idx = all_rows.index(start_row)
    
    for row in all_rows[start_idx + 1:]:
        # Stop if we hit another label
        if row.find('span', class_='cardLabel'):
            break
            
        row_text = row.get_text(strip=True)
        if row_text and row_text != '-':
            items.append(row_text)
    
    return items

def extract_examples_from_row(start_row, all_rows) -> List[str]:
    """
    Extract examples from list items in subsequent rows.
    """
    examples = []
    start_idx = all_rows.index(start_row)
    
    for row in all_rows[start_idx:]:
        # Look for ul/li elements
        ul_elements = row.find_all('ul')
        for ul in ul_elements:
            li_elements = ul.find_all('li')
            for li in li_elements:
                example_text = li.get_text(strip=True)
                if example_text:
                    examples.append(example_text)
        
        # Stop if we hit another label (unless it's the first row)
        if row != start_row and row.find('span', class_='cardLabel'):
            break
    
    return examples

def clean_html_content(content: str) -> str:
    """
    Clean HTML content to plain text while preserving structure.
    """
    if not content:
        return ""
    
    # Remove extra whitespace but preserve some structure
    content = re.sub(r'\s+', ' ', content)
    # Remove any remaining HTML artifacts
    content = re.sub(r'<[^>]+>', '', content)
    # Clean up common HTML entities
    content = content.replace('&nbsp;', ' ')
    content = content.replace('&amp;', '&')
    content = content.replace('&lt;', '<')
    content = content.replace('&gt;', '>')
    content = content.replace('&quot;', '"')
    return content.strip()

def clean_class_reference(text: str) -> str:
    """
    Clean class reference text by removing duplicate class IDs and formatting properly.
    Example: "E1CRM EntityE1" -> "E1 CRM Entity"
    """
    if not text:
        return ""
    
    # Pattern to match class ID followed by class name followed by duplicate class ID
    # E.g., "E1CRM EntityE1" or "E41AppellationE41"
    pattern = r'^(E\d+)([A-Z][^E]*?)\1$'
    match = re.match(pattern, text)
    
    if match:
        class_id = match.group(1)
        class_name = match.group(2).strip()
        return f"{class_id} {class_name}"
    
    # If no match, try to find just the duplicate ID at the end
    # Pattern: "E1CRM EntityE1" where the end E1 is duplicate
    pattern2 = r'^(E\d+)(.+?)(E\d+)$'
    match2 = re.match(pattern2, text)
    
    if match2:
        start_id = match2.group(1)
        middle_text = match2.group(2).strip()
        end_id = match2.group(3)
        
        # If start and end IDs are the same, remove the duplicate
        if start_id == end_id:
            return f"{start_id} {middle_text}"
    
    # If no pattern matches, return original text
    return text

def resolve_class_uri(fragment_uri: str) -> str:
    """
    Convert fragment URI to full CIDOC-CRM URI.
    Example: "#E52" -> "http://www.cidoc-crm.org/cidoc-crm/E52_Time-Span"
    """
    if not fragment_uri or not fragment_uri.startswith('#'):
        return fragment_uri
    
    class_id = fragment_uri[1:]  # Remove the '#'
    
    # Map class IDs to their full names for URI construction
    class_name_map = {
        'E1': 'CRM_Entity',
        'E2': 'Temporal_Entity', 
        'E3': 'Condition_State',
        'E4': 'Period',
        'E5': 'Event',
        'E6': 'Destruction',
        'E7': 'Activity',
        'E8': 'Acquisition',
        'E9': 'Move',
        'E10': 'Transfer_of_Custody',
        'E11': 'Modification',
        'E12': 'Production',
        'E13': 'Attribute_Assignment',
        'E14': 'Condition_Assessment',
        'E15': 'Identifier_Assignment',
        'E16': 'Measurement',
        'E17': 'Type_Assignment',
        'E18': 'Physical_Thing',
        'E19': 'Physical_Object',
        'E20': 'Biological_Object',
        'E21': 'Person',
        'E22': 'Human-Made_Object',
        'E24': 'Physical_Human-Made_Thing',
        'E25': 'Human-Made_Feature',
        'E26': 'Physical_Feature',
        'E27': 'Site',
        'E28': 'Conceptual_Object',
        'E29': 'Design_or_Procedure',
        'E30': 'Right',
        'E31': 'Document',
        'E32': 'Authority_Document',
        'E33': 'Linguistic_Object',
        'E34': 'Inscription',
        'E35': 'Title',
        'E36': 'Visual_Item',
        'E37': 'Mark',
        'E39': 'Actor',
        'E41': 'Appellation',
        'E42': 'Identifier',
        'E52': 'Time-Span',
        'E53': 'Place',
        'E54': 'Dimension',
        'E55': 'Type',
        'E56': 'Language',
        'E57': 'Material',
        'E58': 'Measurement_Unit',
        'E59': 'Primitive_Value',
        'E60': 'Number',
        'E61': 'Time_Primitive',
        'E62': 'String',
        'E63': 'Beginning_of_Existence',
        'E64': 'End_of_Existence',
        'E65': 'Creation',
        'E66': 'Formation',
        'E67': 'Birth',
        'E68': 'Dissolution',
        'E69': 'Death',
        'E70': 'Thing',
        'E71': 'Human-Made_Thing',
        'E72': 'Legal_Object',
        'E73': 'Information_Object',
        'E74': 'Group',
        'E77': 'Persistent_Item',
        'E78': 'Curated_Holding',
        'E79': 'Part_Addition',
        'E80': 'Part_Removal',
        'E81': 'Transformation',
        'E83': 'Type_Creation',
        'E85': 'Joining',
        'E86': 'Leaving',
        'E87': 'Curation_Activity',
        'E89': 'Propositional_Object',
        'E90': 'Symbolic_Object',
        'E92': 'Spacetime_Volume',
        'E93': 'Presence',
        'E94': 'Space_Primitive',
        'E95': 'Spacetime_Primitive',
        'E96': 'Purchase',
        'E97': 'Monetary_Amount',
        'E98': 'Currency',
        'E99': 'Product_Type'
    }
    
    if class_id in class_name_map:
        class_name = class_name_map[class_id]
        return f"http://www.cidoc-crm.org/cidoc-crm/{class_id}_{class_name}"
    
    # Fallback: just use the class ID
    return f"http://www.cidoc-crm.org/cidoc-crm/{class_id}"

def parse_cramped_superclass_text(text: str) -> List[str]:
    """
    Parse cramped superclass text and split it into individual class references.
    Example: "E7ActivityE63Beginning of ExistenceE64End of ExistenceE7E63E64" 
    -> ["E7 Activity", "E63 Beginning of Existence", "E64 End of Existence"]
    """
    if not text or text == '-':
        return []
    
    # Remove trailing duplicate class/property IDs first (like E7E63E64 or P1P2P3 at the end)
    text_clean = re.sub(r'([EP]\d+)+$', '', text)
    
    # More precise regex: match E or P followed by digits, then capture text until next E/P+digits
    # This avoids matching words that start with E or P but aren't class/property IDs
    pattern = r'([EP]\d+)([A-Za-z\s\-]*?)(?=[EP]\d+|$)'
    matches = re.findall(pattern, text_clean)
    
    classes = []
    seen_classes = set()  # To avoid duplicates
    
    for class_id, class_name in matches:
        # Clean up the class name
        class_name = class_name.strip()
        
        # Remove any remaining class/property IDs that got captured
        class_name = re.sub(r'[EP]\d+', '', class_name).strip()
        
        if class_name:
            # Add spaces before capital letters for better readability
            class_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', class_name)
            full_class = f"{class_id} {class_name}"
        else:
            full_class = class_id
            
        # Avoid duplicates
        if full_class not in seen_classes:
            classes.append(full_class)
            seen_classes.add(full_class)
    
    return classes

def clean_superproperty_text(text: str) -> str:
    """
    Clean superproperty text by properly formatting class references and property names.
    Example: "E1CRM Entity.P48has preferred identifier (is preferred identifier of):E42IdentifierE71Human-Made Thing.P102has title (is title of):E35TitleE53Place.P168place is defined by (defines place):E94Space PrimitiveE92Spacetime Volume.P169ispacetime volume is defined by (defines spacetime volume):E95Spacetime PrimitiveE52Time-Span.P170itime is defined by (defines time):E61Time PrimitiveP48P102P168P169iP170i"
    """
    if not text:
        return ""
    
    # Split by property IDs (P followed by digits)
    parts = re.split(r'(P\d+[i]?)', text)
    
    cleaned_parts = []
    for part in parts:
        if not part:
            continue
            
        # If it's a property ID, keep it as is
        if re.match(r'^P\d+[i]?$', part):
            cleaned_parts.append(part)
        else:
            # Clean class references in this part
            # Replace patterns like "E1CRM EntityE1" with "E1 CRM Entity"
            part = re.sub(r'(E\d+)([A-Z][^E]*?)\1', r'\1 \2', part)
            # Clean up any remaining formatting issues
            part = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)  # Add spaces before capitals
            part = re.sub(r'\s+', ' ', part)  # Normalize whitespace
            cleaned_parts.append(part.strip())
    
    result = ''.join(cleaned_parts)
    
    # Final cleanup - remove trailing property IDs that are just concatenated
    result = re.sub(r'(P\d+[i]?)+$', '', result)
    
    return result.strip()

def debug_table_structure(table, entity_id: str):
    """
    Debug function to print table structure for troubleshooting.
    """
    print(f"\n=== DEBUG TABLE STRUCTURE FOR {entity_id} ===")
    rows = table.find_all('tr')
    for i, row in enumerate(rows):
        print(f"Row {i}:")
        # Look for card labels
        card_label = row.find('span', class_='cardLabel')
        if card_label:
            print(f"  Label: {card_label.get_text(strip=True)}")
        
        # Print cell contents
        cells = row.find_all(['td', 'th'])
        for j, cell in enumerate(cells):
            cell_text = cell.get_text(strip=True)[:100]  # Limit length
            print(f"  Cell {j}: {cell_text}")
        print()

def parse_properties_from_tables(soup: BeautifulSoup) -> Dict:
    """
    Parse properties from HTML table structure with better debugging.
    """
    properties = {}
    
    # Find all property spans
    prop_spans = soup.find_all('span', class_='prop')
    print(f"Found {len(prop_spans)} property spans")
    
    for prop_span in prop_spans:
        prop_id = prop_span.get('id', '')
        if not prop_id.startswith('P'):
            continue
            
        # Get the property name from the span text
        prop_text = prop_span.get_text(strip=True)
        
        # Extract property name (everything after the ID)
        prop_name_match = re.match(rf'{re.escape(prop_id)}\s+(.+)', prop_text)
        prop_name = prop_name_match.group(1) if prop_name_match else prop_text
        
        # Find the containing table structure
        prop_table = find_parent_table(prop_span)
        if prop_table:
            # Uncomment for debugging specific properties
            # if prop_id in ['P1', 'P74']:  # Debug specific properties
            #     debug_table_structure(prop_table, prop_id)
                
            property_data = parse_property_from_table(prop_id, prop_name, prop_table)
            if property_data:
                properties[prop_id] = property_data
                print(f"Parsed {prop_id}: {prop_name}")
    
    return properties

def parse_classes_from_tables(soup: BeautifulSoup) -> Dict:
    """
    Parse classes from HTML table structure with better debugging.
    """
    classes = {}
    
    # Find all class spans
    class_spans = soup.find_all('span', class_='cls')
    print(f"Found {len(class_spans)} class spans")
    
    for class_span in class_spans:
        class_id = class_span.get('id', '')
        if not class_id.startswith('E'):
            continue
            
        # Get the class name from the span text
        class_text = class_span.get_text(strip=True)
        
        # Extract class name (everything after the ID)
        class_name_match = re.match(rf'{re.escape(class_id)}\s+(.+)', class_text)
        class_name = class_name_match.group(1) if class_name_match else class_text
        
        # Remove "(show all properties)" if present
        class_name = re.sub(r'\s*\(show all properties\)\s*$', '', class_name)
        
        # Find the containing table structure
        class_table = find_parent_table(class_span)
        if class_table:
            class_data = parse_class_from_table(class_id, class_name, class_table)
            if class_data:
                classes[class_id] = class_data
                print(f"Parsed {class_id}: {class_name}")
    
    return classes

def save_parsed_data(data: Dict, filename: str = "cidoc_crm_parsed.json"):
    """
    Save parsed CIDOC-CRM data to JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved parsed CIDOC-CRM data to {filename}")
        print(f"📊 Properties: {len(data.get('properties', {}))}")
        print(f"📊 Classes: {len(data.get('classes', {}))}")
        
        # Show sample of what was parsed
        if data.get('properties'):
            sample_prop = next(iter(data['properties'].values()))
            print(f"\n📋 Sample Property: {sample_prop['id']} - {sample_prop['name']}")
            if sample_prop['scope_note']:
                print(f"   Scope: {sample_prop['scope_note'][:100]}...")
                
        if data.get('classes'):
            sample_class = next(iter(data['classes'].values()))
            print(f"\n📋 Sample Class: {sample_class['id']} - {sample_class['name']}")
            if sample_class['scope_note']:
                print(f"   Scope: {sample_class['scope_note'][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def main():
    """
    Main function to parse CIDOC-CRM from the official URL.
    """
    print("🚀 Parsing CIDOC-CRM from official URL...")
    
    url = "https://cidoc-crm.org/html/cidoc_crm_v7.1.3.html"
    parsed_data = parse_cidoc_html_from_url(url)
    
    if parsed_data:
        success = save_parsed_data(parsed_data)
        return parsed_data if success else None
    else:
        print("❌ Failed to parse CIDOC-CRM data")
        return None

def parse_local_file(file_path: str):
    """
    Parse CIDOC-CRM from a local HTML file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parsed_data = parse_cidoc_html(html_content)
        save_parsed_data(parsed_data, "cidoc_crm_local_parsed.json")
        return parsed_data
    
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error parsing local file: {e}")
        return None

if __name__ == "__main__":
    main()