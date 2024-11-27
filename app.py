import streamlit as st
import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define data classes
@dataclass
class MedicineEntry:
    concept_id: str
    pt: str
    sdd_class: str  # 'T' for Trade names, others for Generic

class UnitCategories:
    def __init__(self):
        # Basic measurement units
        self.WEIGHT_UNITS = {
            'mg': 'milligram',
            'mgs': 'milligram',
            'milligram': 'milligram',
            'g': 'gram',
            'gm': 'gram',
            'gram': 'gram',
            'grams': 'gram',
            'mcg': 'microgram',
            'µg': 'microgram',
            'ug': 'microgram',
            'micrograms': 'microgram',
            'ng': 'nanogram',
            'kg': 'kilogram'
        }

        self.VOLUME_UNITS = {
            'ml': 'milliliter',
            'ml.': 'milliliter',
            'mls': 'milliliter',
            'millilitre': 'milliliter',
            'milliliters': 'milliliter',
            'cc': 'milliliter',
            'l': 'liter',
            'L': 'liter',
            'lit': 'liter',
            'ltr': 'liter'
        }

        self.INTERNATIONAL_UNITS = {
            'iu': 'unit',
            'i.u': 'unit',
            'i.u.': 'unit',
            'international unit': 'unit',
            'international units': 'unit',
            'u': 'unit',
            'units': 'unit'
        }

        # Time-based patterns
        self.TIME_UNITS = {
            'hr': 'hour',
            'hrs': 'hour',
            'hour': 'hour',
            'hours': 'hour',
            'h': 'hour',
            'd': 'day',
            'day': 'day',
            'days': 'day',
            'wk': 'week',
            'week': 'week',
            'weeks': 'week'
        }

        # Delivery forms
        self.DELIVERY_FORMS = {
            # Oral solid forms
            'tabs': 'tablet',
            'tab': 'tablet',
            'tablet': 'tablet',
            'tablets': 'tablet',
            'cap': 'capsule',
            'caps': 'capsule',
            'capsule': 'capsule',
            'capsules': 'capsule',
            'sachet': 'sachet',
            'sachets': 'sachet',
            'powder': 'powder',
            'powders': 'powder',
            'granules': 'granules',
            'gran': 'granules',
            'lozenge': 'lozenge',
            'lozenges': 'lozenge',
            'sachets': 'sachet',
            'chewable': 'chewable',
            'dispersible': 'dispersible',
            'Sachets': 'sachet',
            
            # Oral liquid forms
            'syrup': 'syrup',
            'syr': 'syrup',
            'suspension': 'suspension',
            'susp': 'suspension',
            'solution': 'solution',
            'soln': 'solution',
            'sol': 'solution',
            'elixir': 'elixir',
            'mixture': 'mixture',
            'mixt': 'mixture',
            'drops': 'drops',
            'drop': 'drops',
            'gtt': 'drops',

            # Inhalation forms
            'inhaler': 'inhaler',
            'inh': 'inhaler',
            'rotacap': 'rotacap',
            'rotahaler': 'rotahaler',
            'mdi': 'metered dose inhaler',
            'dpi': 'dry powder inhaler',
            'nebuliser': 'nebuliser',
            'nebulizer': 'nebuliser',
            'neb': 'nebuliser',
            'respirator': 'respirator',
            'aerosol': 'aerosol',
            
            # Topical forms
            'cream': 'cream',
            'crm': 'cream',
            'ointment': 'ointment',
            'oint': 'ointment',
            'gel': 'gel',
            'lotion': 'lotion',
            'lot': 'lotion',
            'paste': 'paste',
            'patch': 'patch',
            'patches': 'patch',
            'plaster': 'plaster',
            'dressing': 'dressing',
            
            # Suppositories and pessaries
            'supp': 'suppository',
            'supps': 'suppository',
            'suppository': 'suppository',
            'suppositories': 'suppository',
            'pess': 'pessary',
            'pessary': 'pessary',
            'pessaries': 'pessary',
            
            # Injectable forms
            'injection': 'injection',
            'inj': 'injection',
            'amp': 'ampoule',
            'amps': 'ampoule',
            'ampoule': 'ampoule',
            'ampoules': 'ampoule',
            'vial': 'vial',
            'vials': 'vial',
            'pfs': 'prefilled syringe',
            'prefilled syringe': 'prefilled syringe',
            'syringe': 'syringe',
            'cartridge': 'cartridge',
            'cart': 'cartridge',
            
            # Nasal forms
            'nasal spray': 'nasal spray',
            'nasal drops': 'nasal drops',
            'nasal': 'nasal',
            
            # Eye/Ear forms
            'eye drops': 'eye drops',
            'ear drops': 'ear drops',
            'eye ointment': 'eye ointment',
            'ophthalmic': 'ophthalmic',
            'oph': 'ophthalmic',
            'otic': 'otic',
            
            # Sprays
            'spray': 'spray',
            'sprays': 'spray',
            'puff': 'puff',
            'puffs': 'puff',
            
            # Other forms
            'implant': 'implant',
            'device': 'device',
            'foam': 'foam',
            'wash': 'wash',
            'shampoo': 'shampoo',
            'soap': 'soap',
            'spirit': 'spirit',
            'tincture': 'tincture',
            'paint': 'paint'
        }

        # Concentration patterns (composed units)
        self.CONCENTRATION_PATTERNS = [
            (r'mg/ml', 'milligram per milliliter'),
            (r'mg/mL', 'milligram per milliliter'),
            (r'mg/l', 'milligram per liter'),
            (r'mg/L', 'milligram per liter'),
            (r'g/l', 'gram per liter'),
            (r'g/L', 'gram per liter'),
            (r'mcg/ml', 'microgram per milliliter'),
            (r'µg/ml', 'microgram per milliliter'),
            (r'ug/ml', 'microgram per milliliter'),
            (r'ng/ml', 'nanogram per milliliter'),
            (r'iu/ml', 'unit per milliliter'),
            (r'i\.u\./ml', 'unit per milliliter'),
            (r'u/ml', 'unit per milliliter'),
            (r'mmol/l', 'millimole per liter')
        ]

def extract_dosage(text: str) -> List[Dict[str, Any]]:
    """
    Enhanced dosage extraction that handles complex pharmaceutical expressions.
    Returns a list of dictionaries containing dosage information.
    """
    units = UnitCategories()
    results = []
    
    # Create pattern for matching dosage expressions
    value_pattern = r'(\d+(?:\.\d+)?)'
    
    # Build unit patterns
    basic_units = '|'.join(list(units.WEIGHT_UNITS.keys()) + 
                          list(units.VOLUME_UNITS.keys()) + 
                          list(units.INTERNATIONAL_UNITS.keys()))
    time_units = '|'.join(units.TIME_UNITS.keys())
    delivery_forms = '|'.join(units.DELIVERY_FORMS.keys())
    
    # Define patterns
    patterns = [
        # Match time-based dosages (e.g., 15mg/16hr)
        (rf'{value_pattern}\s*({basic_units})\s*(?:/|per)\s*{value_pattern}\s*({time_units})', 'time_based'),
        
        # Match concentrations (e.g., 10mg/ml)
        (rf'{value_pattern}\s*({"|".join(pattern[0] for pattern in units.CONCENTRATION_PATTERNS)})', 'concentration'),
        
        # Match dosage with delivery form (e.g., 25mg tablet, 100mcg inhaler)
        (rf'{value_pattern}\s*({basic_units})\s*({delivery_forms})', 'dosage_form'),
        
        # Match simple dosages (e.g., 25mg)
        (rf'{value_pattern}\s*({basic_units})', 'simple'),
        
        # Match delivery units with count (e.g., 2 tablets, 1 sachet)
        (rf'{value_pattern}\s*({delivery_forms})', 'delivery'),
        
        # Match percentage strength (e.g., 2% w/v)
        (rf'{value_pattern}\s*%\s*(?:w/v|w/w)?', 'percentage')
    ]
    
    def normalize_unit(unit_text: str) -> str:
        """Normalize unit to standard form"""
        for unit_dict in [units.WEIGHT_UNITS, units.VOLUME_UNITS, 
                         units.INTERNATIONAL_UNITS, units.DELIVERY_FORMS]:
            if unit_text.lower() in unit_dict:
                return unit_dict[unit_text.lower()]
        
        # Check concentration patterns
        for pattern, normalized in units.CONCENTRATION_PATTERNS:
            if re.match(pattern, unit_text, re.IGNORECASE):
                return normalized
        
        return unit_text.lower()

    def normalize_time_unit(unit_text: str) -> str:
        """Normalize time unit to standard form"""
        return units.TIME_UNITS.get(unit_text.lower(), unit_text.lower())

    def normalize_delivery_form(form_text: str) -> str:
        """Normalize delivery form to standard form"""
        return units.DELIVERY_FORMS.get(form_text.lower(), form_text.lower())

    # Process each pattern
    for pattern, pattern_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                groups = match.groups()
                dosage_info = {}

                # Always extract value
                dosage_info['value'] = float(groups[0])

                if pattern_type == 'time_based':
                    dosage_info['unit'] = normalize_unit(groups[1])
                    dosage_info['period'] = float(groups[2])
                    dosage_info['period_unit'] = normalize_time_unit(groups[3])
                
                elif pattern_type == 'concentration':
                    dosage_info['unit'] = normalize_unit(groups[1])
                
                elif pattern_type == 'dosage_form':
                    dosage_info['unit'] = normalize_unit(groups[1])
                    dosage_info['delivery_form'] = normalize_delivery_form(groups[2])
                
                elif pattern_type == 'simple':
                    dosage_info['unit'] = normalize_unit(groups[1])
                
                elif pattern_type == 'delivery':
                    dosage_info['delivery_form'] = normalize_delivery_form(groups[1])
                
                elif pattern_type == 'percentage':
                    dosage_info['unit'] = 'percent'

                results.append(dosage_info)

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to process match in pattern {pattern_type}: {e}")
                continue

    return results

def calculate_dosage_similarity(query_dosages: List[Dict], reference_dosages: List[Dict]) -> float:
    """
    Calculate similarity score between dosages.
    Returns a score between 0 and 1, where 1 indicates perfect match.
    """
    if not query_dosages or not reference_dosages:
        return 0.5  # Default score if no dosages found
    
    max_score = 0
    for query_dosage in query_dosages:
        for ref_dosage in reference_dosages:
            score = 0
            
            # Compare units
            if query_dosage.get('unit') == ref_dosage.get('unit'):
                score += 0.3
                
                # Compare values if units match and neither value is zero
                query_value = query_dosage.get('value', 0)
                ref_value = ref_dosage.get('value', 0)
                
                if query_value > 0 and ref_value > 0:
                    value_ratio = min(query_value, ref_value) / max(query_value, ref_value)
                    score += 0.3 * value_ratio
                elif query_value == ref_value == 0:
                    # If both values are zero, consider it a perfect match
                    score += 0.3
                else:
                    # One value is zero and the other isn't - partial match
                    score += 0.15
            
            # Compare delivery forms if present
            query_form = query_dosage.get('delivery_form')
            ref_form = ref_dosage.get('delivery_form')
            if query_form and ref_form and query_form == ref_form:
                score += 0.2
            
            # Compare time periods if present
            if 'period' in query_dosage and 'period' in ref_dosage:
                if query_dosage.get('period_unit') == ref_dosage.get('period_unit'):
                    score += 0.1
                    
                    query_period = query_dosage.get('period', 0)
                    ref_period = ref_dosage.get('period', 0)
                    
                    if query_period > 0 and ref_period > 0:
                        period_ratio = min(query_period, ref_period) / max(query_period, ref_period)
                        score += 0.1 * period_ratio
                    elif query_period == ref_period == 0:
                        score += 0.1
                    else:
                        score += 0.05
            
            max_score = max(max_score, score)
    
    return max_score


def preprocess_text(text: str) -> str:
    """Preprocess text for matching"""
    text = text.lower()
    text = re.sub(r'[\(\)\[\],]', '', text)  # Remove brackets and commas
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = text.strip()
    return text

def remove_packaging_info(text: str) -> str:
    """Remove packaging information from text"""
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\b(box of|pack of|bottle of|vial of)\b.*', '', text, flags=re.IGNORECASE)
    return text.strip()

def preprocess_query(query: str) -> str:
    """Preprocess query text"""
    query = remove_packaging_info(query)
    query = preprocess_text(query)
    return query

def calculate_similarity_scores(query: str, reference_texts: List[str]) -> List[float]:
    """Calculate TF-IDF cosine similarity scores"""
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([query] + reference_texts)
        query_vector = vectors[0]
        ref_vectors = vectors[1:]
        cosine_similarities = cosine_similarity(query_vector, ref_vectors).flatten()
        return cosine_similarities
    except Exception as e:
        logger.error(f"Error calculating similarity scores: {e}")
        return [0.0] * len(reference_texts)

def calculate_fuzzy_scores(query: str, reference_texts: List[str]) -> List[float]:
    """Calculate fuzzy matching scores"""
    try:
        scores = [fuzz.token_set_ratio(query, ref) / 100 for ref in reference_texts]
        return scores
    except Exception as e:
        logger.error(f"Error calculating fuzzy scores: {e}")
        return [0.0] * len(reference_texts)

def find_best_match(query: str, entries: List[MedicineEntry]) -> Tuple[MedicineEntry, float]:
    """Find best matching medicine entry"""
    if not entries:  # Add check for empty entries list
        return None, 0.0
        
    preprocessed_query = preprocess_query(query)
    query_dosages = extract_dosage(query)
    
    reference_texts = [preprocess_query(entry.pt) for entry in entries]
    reference_dosages = [extract_dosage(entry.pt) for entry in entries]
    
    # Calculate similarity scores
    cosine_scores = calculate_similarity_scores(preprocessed_query, reference_texts)
    fuzzy_scores = calculate_fuzzy_scores(preprocessed_query, reference_texts)
    dosage_scores = [calculate_dosage_similarity(query_dosages, ref_dosage) 
                    for ref_dosage in reference_dosages]
    
    # Combine scores with weights
    combined_scores = [0.35 * cos + 0.35 * fuzz + 0.3 * dos 
                      for cos, fuzz, dos in zip(cosine_scores, fuzzy_scores, dosage_scores)]
    
    best_index = np.argmax(combined_scores)
    best_match = entries[best_index]
    best_score = combined_scores[best_index]
    
    return best_match, best_score

def find_best_trade_match(query: str, entries: List[MedicineEntry]) -> Tuple[MedicineEntry, float]:
    """Find best matching trade medicine"""
    trade_entries = [entry for entry in entries if entry.sdd_class and 'T' in entry.sdd_class]
    if not trade_entries:
        return None, 0.0
    return find_best_match(query, trade_entries)

def find_best_generic_match(query: str, entries: List[MedicineEntry], trade_match: Optional[MedicineEntry] = None) -> Tuple[MedicineEntry, float]:
    """
    Find best matching generic medicine using the trade match result if available.
    
    Args:
        query (str): The original medicine name to match
        entries (List[MedicineEntry]): List of all medicine entries
        trade_match (Optional[MedicineEntry]): The best matching trade medicine, if found
    
    Returns:
        Tuple[MedicineEntry, float]: Best matching generic medicine and its similarity score
    """
    # Filter generic entries with improved validation
    generic_entries = [
        entry for entry in entries 
        if entry.sdd_class and 
        'T' not in entry.sdd_class.upper() and 
        entry.pt and 
        isinstance(entry.pt, str) and 
        len(entry.pt.strip()) > 0
    ]
    
    if not generic_entries:
        return None, 0.0
    
    # If we have a trade match, use it as the primary query
    if trade_match and trade_match.pt:
        primary_query = trade_match.pt
    else:
        primary_query = query
        
    preprocessed_query = preprocess_query(primary_query)
    query_dosages = extract_dosage(primary_query)
    
    # Process query ingredients
    query_parts = re.split(r'\s*[+/]\s*|\s+(?:and|with)\s+', preprocessed_query.lower())
    query_base = query_parts[0]  # Primary ingredient/name
    
    best_match = None
    best_score = 0.0
    
    for entry in generic_entries:
        entry_text = preprocess_query(entry.pt)
        entry_dosages = extract_dosage(entry.pt)
        
        # Base text similarity calculations
        cosine_score = calculate_similarity_scores(preprocessed_query, [entry_text])[0]
        fuzzy_score = calculate_fuzzy_scores(preprocessed_query, [entry_text])[0]
        dosage_score = calculate_dosage_similarity(query_dosages, entry_dosages)
        
        # Enhanced scoring based on medicine name patterns
        name_pattern_score = 0.0
        entry_parts = re.split(r'\s*[+/]\s*|\s+(?:and|with)\s+', entry_text.lower())
        
        # Check for exact matches in primary ingredients
        if query_base == entry_parts[0]:
            name_pattern_score = 1.0
        # Check for partial matches in primary ingredients
        elif query_base in entry_parts[0] or entry_parts[0] in query_base:
            name_pattern_score = 0.8
        # Check for matches in other parts
        elif any(part in entry_text.lower() for part in query_parts):
            name_pattern_score = 0.6
        
        # Dosage pattern matching bonus
        dosage_pattern_bonus = 0.0
        if query_dosages and entry_dosages:
            query_values = {d.get('value', 0) for d in query_dosages}
            entry_values = {d.get('value', 0) for d in entry_dosages}
            if query_values.intersection(entry_values):
                dosage_pattern_bonus = 0.1
        
        # Trade match bonus - give higher weight if we're matching from a trade name
        trade_match_bonus = 0.1 if trade_match else 0.0
        
        # Calculate final combined score with adjusted weights
        combined_score = (
            0.25 * cosine_score +      # Text similarity (reduced weight)
            0.20 * fuzzy_score +       # Fuzzy matching (reduced weight)
            0.25 * dosage_score +      # Dosage similarity
            0.15 * name_pattern_score + # Name pattern matching
            0.05 * dosage_pattern_bonus + # Exact dosage match bonus
            0.10 * trade_match_bonus    # Trade match bonus
        )
        
        # Update best match if current score is higher
        if combined_score > best_score:
            best_score = combined_score
            best_match = entry
    
    return best_match, best_score

@st.cache_data
def load_reference_data(filepath: str) -> List[MedicineEntry]:
    """Load and cache reference data"""
    try:
        df = pd.read_csv(filepath)
        required_columns = {'ConceptId', 'pt', 'sddClass'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        entries = [
            MedicineEntry(
                concept_id=str(row['ConceptId']),
                pt=str(row['pt']),
                sdd_class=str(row['sddClass'])
            )
            for _, row in df.iterrows()
        ]
        return entries
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        raise

def main():
    """Main Streamlit application"""
    st.title("Enhanced Medicine Concept Mapper")
    st.write("Match medicine names to standardized concepts using advanced text processing techniques.")

    # Add threshold setting in sidebar
    score_threshold = st.sidebar.slider(
        "Matching Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.43,
        help="Matches with scores below this threshold will be treated as no match"
    )

    # Load reference data
    data_file = 'data.csv'
    if not os.path.exists(data_file):
        st.error(f"Data file '{data_file}' not found. Please upload the reference CSV file.")
        uploaded_ref = st.file_uploader("Upload Reference CSV", type=['csv'])
        if uploaded_ref is not None:
            with open("data_uploaded.csv", "wb") as f:
                f.write(uploaded_ref.getbuffer())
            data_file = "data_uploaded.csv"
    
    try:
        entries = load_reference_data(data_file)
    except Exception as e:
        st.error(f"Error loading reference data: {e}")
        return

    # User interface
    mode = st.sidebar.selectbox("Select Mode", ["Single Query", "Batch Processing"])

    if mode == "Single Query":
        query = st.text_input("Enter the medicine name:")
        if st.button("Find Match"):
            if query:
                with st.spinner("Finding matches..."):
                    best_trade_match, best_trade_score = find_best_trade_match(query, entries)
                    best_generic_match, best_generic_score = find_best_generic_match(query, entries,best_trade_match)

                    # Display query analysis
                    st.write("### Query Analysis")
                    query_dosages = extract_dosage(query)
                    if query_dosages:
                        st.write("Detected dosages:")
                        for dosage in query_dosages:
                            st.write(dosage)

                    # Display trade match results
                    st.write("### Best Trade Match:")
                    if best_trade_match and best_trade_score >= score_threshold:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Name:**", best_trade_match.pt)
                            st.write("**Concept ID:**", best_trade_match.concept_id)
                        with col2:
                            st.write("**sddClass:**", best_trade_match.sdd_class)
                            st.write("**Similarity Score:**", f"{best_trade_score:.2f}")
                        
                        # Display trade match dosage analysis
                        trade_dosages = extract_dosage(best_trade_match.pt)
                        if trade_dosages:
                            st.write("Detected dosages:")
                            for dosage in trade_dosages:
                                st.write(dosage)
                    else:
                        st.write(f"No Trade match found with score >= {score_threshold}")
                        if best_trade_score > 0:
                            st.write(f"(Best score was: {best_trade_score:.2f})")

                    # Display generic match results
                    st.write("### Best Generic Match:")
                    if best_generic_match and best_generic_score >= score_threshold:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Name:**", best_generic_match.pt)
                            st.write("**Concept ID:**", best_generic_match.concept_id)
                        with col2:
                            st.write("**sddClass:**", best_generic_match.sdd_class)
                            st.write("**Similarity Score:**", f"{best_generic_score:.2f}")
                        
                        # Display generic match dosage analysis
                        generic_dosages = extract_dosage(best_generic_match.pt)
                        if generic_dosages:
                            st.write("Detected dosages:")
                            for dosage in generic_dosages:
                                st.write(dosage)
                    else:
                        st.write(f"No Generic match found with score >= {score_threshold}")
                        if best_generic_score > 0:
                            st.write(f"(Best score was: {best_generic_score:.2f})")
            else:
                st.warning("Please enter a medicine name.")

    elif mode == "Batch Processing":
        st.write("### Batch Processing")
        st.write("Upload a CSV file with a 'MedicineName' column to process multiple queries at once.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df_queries = pd.read_csv(uploaded_file)
                
                if 'MedicineName' not in df_queries.columns:
                    st.error("CSV file must contain a 'MedicineName' column.")
                    return
                
                # Check if DataFrame is empty
                if df_queries.empty:
                    st.error("The uploaded CSV file is empty.")
                    return
                
                with st.spinner("Processing queries..."):
                    results = []
                    total_rows = len(df_queries)
                    progress_bar = st.progress(0)
                    
                    for idx, row in df_queries.iterrows():
                        query = str(row['MedicineName'])  # Convert to string to handle non-string inputs
                        
                        # Skip empty or NaN values
                        if pd.isna(query) or query.strip() == '':
                            continue
                            
                        best_trade_match, best_trade_score = find_best_trade_match(query, entries)
                        best_generic_match, best_generic_score = find_best_generic_match(query, entries, best_trade_match)
                        
                        # Extract dosages for analysis
                        try:
                            query_dosages = extract_dosage(query)
                        except Exception as e:
                            logger.warning(f"Error extracting dosages from query '{query}': {e}")
                            query_dosages = []
                        
                        # Apply threshold to trade match
                        if best_trade_score < score_threshold:
                            trade_result = {
                                'Trade_ConceptId': None,
                                'Trade_Name': None,
                                'Trade_sddClass': None,
                                'Trade_Similarity_Score': best_trade_score,
                                'Trade_Dosages': None
                            }
                        else:
                            try:
                                trade_dosages = extract_dosage(best_trade_match.pt) if best_trade_match else []
                            except Exception as e:
                                logger.warning(f"Error extracting trade dosages: {e}")
                                trade_dosages = []
                                
                            trade_result = {
                                'Trade_ConceptId': best_trade_match.concept_id if best_trade_match else None,
                                'Trade_Name': best_trade_match.pt if best_trade_match else None,
                                'Trade_sddClass': best_trade_match.sdd_class if best_trade_match else None,
                                'Trade_Similarity_Score': best_trade_score,
                                'Trade_Dosages': str(trade_dosages) if trade_dosages else None
                            }

                        # Apply threshold to generic match
                        if best_generic_score < score_threshold:
                            generic_result = {
                                'Generic_ConceptId': None,
                                'Generic_Name': None,
                                'Generic_sddClass': None,
                                'Generic_Similarity_Score': best_generic_score,
                                'Generic_Dosages': None
                            }
                        else:
                            try:
                                generic_dosages = extract_dosage(best_generic_match.pt) if best_generic_match else []
                            except Exception as e:
                                logger.warning(f"Error extracting generic dosages: {e}")
                                generic_dosages = []
                                
                            generic_result = {
                                'Generic_ConceptId': best_generic_match.concept_id if best_generic_match else None,
                                'Generic_Name': best_generic_match.pt if best_generic_match else None,
                                'Generic_sddClass': best_generic_match.sdd_class if best_generic_match else None,
                                'Generic_Similarity_Score': best_generic_score,
                                'Generic_Dosages': str(generic_dosages) if generic_dosages else None
                            }
                        
                        # Combine results
                        result = {
                            'MedicineName': query,
                            'Query_Dosages': str(query_dosages) if query_dosages else None,
                            **trade_result,
                            **generic_result
                        }
                        
                        results.append(result)
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / total_rows)
                    
                    # Check if we have any results
                    if not results:
                        st.warning("No valid results were generated from the input data.")
                        return
                        
                    # Create results DataFrame
                    df_results = pd.DataFrame(results)
                    
                    # Display results
                    st.write("### Results")
                    st.dataframe(df_results)
                    
                    # Provide download option
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='medicine_matching_results.csv',
                        mime='text/csv'
                    )
                    
                    # Display summary statistics with safety checks
                    st.write("### Summary Statistics")
                    total_processed = len(df_results)
                    st.write(f"Total queries processed: {total_processed}")
                    
                    if total_processed > 0:  # Prevent division by zero
                        valid_trade_scores = df_results['Trade_Similarity_Score'][
                            df_results['Trade_Similarity_Score'] >= score_threshold]
                        valid_generic_scores = df_results['Generic_Similarity_Score'][
                            df_results['Generic_Similarity_Score'] >= score_threshold]
                        
                        trade_matches = len(valid_trade_scores)
                        generic_matches = len(valid_generic_scores)
                        
                        st.write(f"Matches found above threshold ({score_threshold}):")
                        st.write(f"- Trade matches: {trade_matches} ({(trade_matches/total_processed*100):.1f}%)")
                        st.write(f"- Generic matches: {generic_matches} ({(generic_matches/total_processed*100):.1f}%)")
                        
                        if trade_matches > 0:
                            st.write(f"Average Trade match score (above threshold): {valid_trade_scores.mean():.2f}")
                        if generic_matches > 0:
                            st.write(f"Average Generic match score (above threshold): {valid_generic_scores.mean():.2f}")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                logger.error(f"Batch processing error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")