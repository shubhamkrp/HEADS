import argparse
import json
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ASR dataset from Excel metadata and JSON transcripts")
    parser.add_argument("--excel-path", type=str, required=True, help="Path to speaker_profiles_with_filelocations.xlsx")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory containing the .wav segment files")
    parser.add_argument("--json-dir", type=str, required=True, help="Directory containing the .json transcript files")
    parser.add_argument("--output-dir", type=str, default="asr_dataset_with_filename", help="Output directory for the HF dataset")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Target sampling rate (default: 16000)")
    return parser.parse_args()

def normalize_filename(filename):
    """
    Standardizes filenames by lowercasing and removing extensions/suffixes.
    Matches the user's requirement for .wav, .json, .txt, and _transcript.
    """
    if filename is None or pd.isna(filename):
        return ""
    
    # Convert to string, lowercase, and strip whitespace
    s = str(filename).lower().strip()
    
    # Remove common extensions
    for ext in [".json", ".wav", ".txt", ".m4a", ".mp3"]:
        if s.endswith(ext):
            s = s[:-len(ext)]
    
    # Remove the _transcript suffix if present
    s = s.replace("_transcript", "")
    
    return s.strip()

def normalize_language(lang):
    """Normalize language strings to match fine-tuning script keys."""
    if not isinstance(lang, str):
        return "english"
    lang = lang.lower().strip()
    if "hindi" in lang:
        return "hindi"
    if "kannada" in lang:
        return "kannada"
    return "english"

def main():
    args = parse_args()
    excel_path = Path(args.excel_path)
    audio_dir = Path(args.audio_dir)
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)

    # 1. Load and Clean Metadata
    print(f"Loading metadata from {excel_path}...")
    # Use engine='openpyxl' for .xlsx files
    df = pd.read_excel(excel_path)
    
    # Apply normalization to the Excel FileName column
    df["base_filename"] = df["FileName"].apply(normalize_filename)
    
    metadata_map = {}
    for _, row in df.iterrows():
        base_name = row["base_filename"]
        if not base_name:
            continue
            
        fold = str(row.get("fold", "train")).lower().strip()
        language = normalize_language(row.get("language", "english"))
        
        # Standardize split names
        if fold in ["val", "validation", "dev"]:
            fold = "dev"
        elif fold in ["test", "testing"]:
            fold = "test"
        else:
            fold = "train"
            
        metadata_map[base_name] = {
            "fold": fold,
            "language": language
        }

    print(f"Loaded metadata for {len(metadata_map)} unique base files.")

    # 2. Collect Samples
    data_splits = {
        "train": {"audio": [], "transcription": [], "language": [], "filename": []},
        "dev": {"audio": [], "transcription": [], "language": [], "filename": []},
        "test": {"audio": [], "transcription": [], "language": [], "filename": []}
    }

    json_files = list(json_dir.glob("*.json"))
    print(f"Processing {len(json_files)} JSON transcripts...")
    
    missing_audio_count = 0
    missing_meta_count = 0
    failed_keys = set()
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Normalize the JSON filename/base_name for lookup
            raw_base_name = data.get("base_name") or json_file.name
            clean_base_name = normalize_filename(raw_base_name)
            
            meta = metadata_map.get(clean_base_name)
            
            if not meta:
                missing_meta_count += 1
                failed_keys.add(clean_base_name)
                continue

            fold = meta["fold"]
            language = meta["language"]

            for segment in data.get("segments", []):
                wav_filename = segment.get("filename")
                # Handle potential None/null values in transcription
                raw_transcript = segment.get("transcription")
                
                if raw_transcript is None or not wav_filename:
                    continue
                
                transcript = str(raw_transcript).strip()
                if not transcript:
                    continue
                    
                audio_path = audio_dir / wav_filename
                
                if not audio_path.exists():
                    missing_audio_count += 1
                    continue
                
                data_splits[fold]["audio"].append(str(audio_path.absolute()))
                data_splits[fold]["transcription"].append(transcript)
                data_splits[fold]["language"].append(language)
                data_splits[fold]["filename"].append(wav_filename)
                
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    print(f"\n--- Processing Summary ---")
    print(f"Skipped {missing_meta_count} JSONs: No matching metadata found.")
    if failed_keys and missing_meta_count > 0:
        print(f"Sample of keys that failed to match: {list(failed_keys)[:5]}")
        print(f"Sample of keys available in Excel: {list(metadata_map.keys())[:5]}")
        
    print(f"Skipped {missing_audio_count} segments: Audio file missing on disk.")

    # 3. Create Hugging Face Dataset
    datasets = {}
    for split_name, split_data in data_splits.items():
        if not split_data["audio"]:
            continue
            
        print(f"Building '{split_name}' split with {len(split_data['audio'])} examples...")
        ds = Dataset.from_dict(split_data)
        # Casting "audio" handles loading the arrays from disk, while "filename" stays a raw string
        ds = ds.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
        datasets[split_name] = ds

    if not datasets:
        print("Error: No data successfully matched. Check filename formats and logs above.")
        return

    dataset_dict = DatasetDict(datasets)

    # 4. Save to Disk
    print(f"Saving dataset to: {output_dir}...")
    dataset_dict.save_to_disk(output_dir)
    print("Success! Dataset is ready.")

if __name__ == "__main__":
    main()