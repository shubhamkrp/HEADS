import os
import json
import re
import pandas as pd
import numpy as np
import jiwer
from pathlib import Path

# Configuration 
JSON_DIR = "transcriptions_gemma3n_ft" ## Update this based on the transcription from gemma3n model (ft, ps, cl, ctc)
# print(JSON_DIR)
METADATA_PATH = "speaker_profiles.xlsx"

# Fairness Score Weights
ALPHA = 0.5
BETA = 0.5  

# Regex to identify and split by speaker labels across languages
SPEAKER_REGEX = re.compile(r'(Doctor:|Patient:|डॉक्टर:|मरीज़:|ವೈದ್ಯ:|ರೋಗಿ:)', re.IGNORECASE)

# Helper Functions
def normalize_filename(filename, is_json=False):
    """Standardizes filenames by lowercasing and removing extensions/suffixes."""
    if filename is None or pd.isna(filename):
        return ""
    s = str(filename).lower().strip()
    for ext in [".json", ".wav", ".txt", ".m4a", ".mp3"]:
        if s.endswith(ext):
            s = s[:-len(ext)]
    s = s.replace("_transcript", "")
    if is_json:
        s = re.sub(r'_\d{4}$', '', s)
    return s.strip()

def map_education(edu_str):
    """Maps education to <Graduate or >=Graduate groups."""
    if pd.isna(edu_str): return np.nan
    edu_str = str(edu_str).lower()
    if any(lvl in edu_str for lvl in ["not formally educated", "primary", "secondary"]):
        return "<Graduate"
    elif any(lvl in edu_str for lvl in ["graduate", "post graduate", "pg", "ug"]):
        return ">=Graduate"
    return np.nan

def standardize_gender(gender_str):
    """Standardizes gender strings to 'M' or 'F'."""
    if pd.isna(gender_str): return np.nan
    g = str(gender_str).strip().upper()
    if g.startswith('M'): return 'M'
    if g.startswith('F'): return 'F'
    return np.nan

def calculate_fs(error_g1, error_g2, alpha=1.0, beta=1.0):
    """Calculates the Fairness Score (FS)."""
    if pd.isna(error_g1) or pd.isna(error_g2):
        return np.nan
    avg_error_rate = (error_g1 + error_g2) / 2.0
    error_disparity = abs(error_g1 - error_g2)
    return -alpha * avg_error_rate - beta * error_disparity

def parse_ground_truth(gt_text):
    """Parses ground truth into a list of words and a mapped list of speakers."""
    if not isinstance(gt_text, str): return [], []
    
    # Split keeping the delimiters
    tokens = SPEAKER_REGEX.split(gt_text)
    gt_words = []
    speaker_map = []
    current_speaker = "Unknown"
    
    for token in tokens:
        token_lower = token.lower().strip()
        if token_lower in ['doctor:', 'डॉक्टर:', 'ವೈದ್ಯ:']:
            current_speaker = 'Doctor'
            continue
        elif token_lower in ['patient:', 'मरीज़:', 'ರೋಗಿ:']:
            current_speaker = 'Patient'
            continue
            
        words = token.split()
        if words:
            gt_words.extend(words)
            speaker_map.extend([current_speaker] * len(words))
            
    return gt_words, speaker_map

def main():
    print("Loading metadata...")
    df_meta = pd.read_excel(METADATA_PATH)
    df_meta['base_filename'] = df_meta['FileName'].apply(lambda x: normalize_filename(x, is_json=False))
    df_meta['edu_group'] = df_meta['patient_education'].apply(map_education)
    df_meta['patient_gender'] = df_meta['patient_gender'].apply(standardize_gender)
    df_meta['doctor_gender'] = df_meta['doctor_gender'].apply(standardize_gender)
    
    print("Processing JSON files and calculating alignments...")
    results = []
    json_files = list(Path(JSON_DIR).glob("*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        base_filename = normalize_filename(data.get("filename", ""), is_json=True)
        lang = data.get("language", "english")
        gt_raw = data.get("ground_truth", "")
        pred_raw = data.get("prediction", "")
        
        # Prepare Text
        gt_words, speaker_map = parse_ground_truth(gt_raw)
        gt_clean_str = " ".join(gt_words)
        
        pred_clean_str = SPEAKER_REGEX.sub("", pred_raw)
        pred_clean_str = re.sub(r'\s+', ' ', pred_clean_str).strip()
        
        # Overall WER
        try:
            overall_wer = jiwer.wer(gt_clean_str, pred_clean_str) if gt_clean_str and pred_clean_str else np.nan
        except ValueError:
            overall_wer = np.nan
            
        # Role-wise WER via Alignment
        doc_wer, pat_wer = np.nan, np.nan
        
        if gt_clean_str and pred_clean_str:
            out = jiwer.process_words(gt_clean_str, pred_clean_str)
            alignments = out.alignments[0]
            
            doc_err, doc_ref = 0, 0
            pat_err, pat_ref = 0, 0
            
            for chunk in alignments:
                c_type = chunk.type
                r_start = chunk.ref_start_idx
                r_end = chunk.ref_end_idx
                
                # Reference Words
                if c_type in ['equal', 'substitute', 'delete']:
                    for i in range(r_start, r_end):
                        speaker = speaker_map[i]
                        if speaker == 'Doctor': doc_ref += 1
                        elif speaker == 'Patient': pat_ref += 1
                
                # Count Errors
                if c_type in ['substitute', 'delete']:
                    for i in range(r_start, r_end):
                        speaker = speaker_map[i]
                        if speaker == 'Doctor': doc_err += 1
                        elif speaker == 'Patient': pat_err += 1
                        
                elif c_type == 'insert':
                    # Assign insertion error to the preceding word's speaker
                    ref_idx = r_start - 1 if r_start > 0 else 0
                    if ref_idx < len(speaker_map):
                        speaker = speaker_map[ref_idx]
                        num_inserted = chunk.hyp_end_idx - chunk.hyp_start_idx
                        if speaker == 'Doctor': doc_err += num_inserted
                        elif speaker == 'Patient': pat_err += num_inserted
            
            doc_wer = doc_err / doc_ref if doc_ref > 0 else np.nan
            pat_wer = pat_err / pat_ref if pat_ref > 0 else np.nan

        results.append({
            "base_filename": base_filename,
            "overall_wer": overall_wer,
            "doctor_wer": doc_wer,
            "patient_wer": pat_wer
        })
        
    df_results = pd.DataFrame(results)
    df_merged = pd.merge(df_results, df_meta, on='base_filename', how='inner')
    
    # CALCULATING METRICS
    print("\n" + "="*40)
    print("             MEDIAN WERs")
    print("="*40)
    
    # Overall
    overall_median = df_merged['overall_wer'].median()
    print(f"1. Overall Median WER: {overall_median:.4f}")
    
    # Language-wise
    print("\n2. Language-wise Median WER:")
    print(df_merged.groupby('language')['overall_wer'].median().to_string())
    
    # Gender-wise
    female_wers, male_wers = [], []
    for _, row in df_merged.iterrows():
        p_gender = row.get('patient_gender')
        if pd.notna(row['patient_wer']):
            if p_gender == 'F': female_wers.append(row['patient_wer'])
            elif p_gender == 'M': male_wers.append(row['patient_wer'])
            
        d_gender = row.get('doctor_gender')
        if pd.notna(row['doctor_wer']):
            if d_gender == 'F': female_wers.append(row['doctor_wer'])
            elif d_gender == 'M': male_wers.append(row['doctor_wer'])

    median_wer_f = np.median(female_wers) if female_wers else np.nan
    median_wer_m = np.median(male_wers) if male_wers else np.nan
    print("\n3. Gender-wise Median WER:")
    print(f"   Female: {median_wer_f:.4f}")
    print(f"   Male:   {median_wer_m:.4f}")
    
    # Role-wise
    median_wer_doc = df_merged['doctor_wer'].median()
    median_wer_pat = df_merged['patient_wer'].median()
    print("\n4. Role-wise Median WER:")
    print(f"   Doctor:  {median_wer_doc:.4f}")
    print(f"   Patient: {median_wer_pat:.4f}")
    
    # Education-wise
    edu_med = df_merged.groupby('edu_group')['patient_wer'].median()
    print("\n5. Education-wise Median WER (Patient only):")
    print(edu_med.to_string())
    
    # FAIRNESS SCORES 
    print("\n" + "="*40)
    print("          FAIRNESS SCORES (FS)")
    print("="*40)
    
    # Gender FS
    fs_gender = calculate_fs(median_wer_f, median_wer_m, ALPHA, BETA)
    print(f"1. Gender FS (Female vs Male):       {fs_gender:.4f}")
    
    # Role FS
    fs_role = calculate_fs(median_wer_pat, median_wer_doc, ALPHA, BETA)
    print(f"2. Role FS (Patient vs Doctor):      {fs_role:.4f}")
    
    # Education FS
    if '<Graduate' in edu_med and '>=Graduate' in edu_med:
        fs_edu = calculate_fs(edu_med['<Graduate'], edu_med['>=Graduate'], ALPHA, BETA)
        print(f"3. Education FS (<Grad vs >=Grad):  {fs_edu:.4f}")
    else:
        print("3. Education FS: Cannot calculate (missing groups in data).")

if __name__ == "__main__":
    main()
