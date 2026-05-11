import os
import json
import torch
from unsloth import FastModel
from peft import PeftModel
from datasets import load_from_disk, Audio

# Configuration & Setup
base_model_name = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"

checkpoint_path = "outputs-cl/final_model"  # Update this based on the ablation gemma3n model (ft, ps, cl, ctc)
dataset_path = "asr_dataset"
prompt_dir = "prompts"
output_dir = "transcriptions_gemma3n_cl"

os.makedirs(output_dir, exist_ok=True)

# Load Prompts
prompt_mapping = [
    ("english", "transcription_prompt_chunks_gemma.txt"),
    ("hindi", "transcription_prompt_chunks_gemma_hindi.txt"),
    ("kannada", "transcription_prompt_chunks_gemma_kannada.txt"),
]

prompts = {}
print("Loading language prompts...")
for lang, fname in prompt_mapping:
    path = os.path.join(prompt_dir, fname)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            prompts[lang] = f.read().strip()
    else:
        print(f"Warning: Prompt file {fname} not found. Using default prompt.")
        prompts[lang] = "Please transcribe this audio."

# Load Base Model, Processor, and Checkpoint
print(f"Loading base model: {base_model_name}...")
model, processor = FastModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=None,
)

print(f"Loading LoRA adapters from checkpoint: {checkpoint_path}...")
model = PeftModel.from_pretrained(model, checkpoint_path)

FastModel.for_inference(model)

# Load and Prepare Test Dataset
print("Loading test dataset...")
dataset = load_from_disk(dataset_path)

if "test" not in dataset:
    raise ValueError("The dataset does not contain a 'test' split.")

test_dataset = dataset["test"]
# Standardize the audio inputs to mono, 16kHz to match training
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

print("Starting inference on the test set...")
model.eval()

results_buffer = []

for i, sample in enumerate(test_dataset):
    # get language
    lang = sample.get('language', 'english').lower()
    
    # Map short codes to full names if needed
    if lang == 'en': lang = 'english'
    elif lang == 'hi': lang = 'hindi'
    elif lang == 'kn': lang = 'kannada'
        
    user_prompt = prompts.get(lang, prompts.get('english'))
    audio_array = sample['audio']['array']
    ground_truth = sample.get('transcription', '')
    filename = sample.get('filename', '')
    
    # chat template
    messages = [
        {
            'role': 'system', 
            'content': [{'type': 'text', 'text': 'You are an assistant that transcribes speech accurately.'}]
        },
        {
            'role': 'user', 
            'content': [
                {'type': 'audio', 'audio': audio_array}, 
                {'type': 'text', 'text': user_prompt}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True, 
        return_tensors='pt'
    ).to(model.device, dtype=torch.bfloat16)
    
    # generate transcription
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, num_beams=6)
        
    decoded_result = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    final_result = ''.join(decoded_result[0].split('\nmodel\n')[-1]).strip()
    
    # prepare JSON payload
    result_data = {
        "filename": filename,
        "language": lang,
        "ground_truth": ground_truth,
        "prediction": final_result
    }
    
    out_file = os.path.join(output_dir, f"sample_{i:04d}.json")
    results_buffer.append((out_file, result_data))
    
    # Save to directory every 50 inferences
    if (i + 1) % 50 == 0:
        for filepath, data in results_buffer:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        results_buffer.clear()
        print(f"Processed and saved {i + 1}/{len(test_dataset)} samples...")

if results_buffer:
    for filepath, data in results_buffer:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    results_buffer.clear()
    print(f"Processed and saved the final {len(test_dataset)} samples...")

print(f"\nInference complete! All transcriptions are saved in the '{output_dir}' directory.")
