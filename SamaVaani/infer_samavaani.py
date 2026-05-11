import os
import json
import torch
from datasets import load_from_disk, Audio
from unsloth import FastModel

# Configuration & Setup
BASE_MODEL_ID = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
CHECKPOINT_DIR = "outputs_ctc/final_model"
DATASET_DIR = "asr_dataset"
OUTPUT_DIR = "transcriptions_cl_ctc"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prevent recompilation limit issues
torch._dynamo.config.cache_size_limit = 32

# Load Prompts per Language
print("Loading language prompts...")
prompt_files = {
    "english": "transcription_prompt_chunks_gemma.txt",
    "hindi": "transcription_prompt_chunks_gemma_hindi.txt",
    "kannada": "transcription_prompt_chunks_gemma_kannada.txt",
}

prompts = {}
for lang, fname in prompt_files.items():
    prompt_path = os.path.join("prompts", fname)
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts[lang] = f.read().strip()
    else:
        print(f"Warning: Could not find {prompt_path}. Using fallback prompt.")
        prompts[lang] = "Please transcribe this audio accurately."

# Load Dataset
print(f"Loading dataset from {DATASET_DIR}...")
dataset = load_from_disk(DATASET_DIR)

if "test" not in dataset:
    raise ValueError("No 'test' split found in the dataset.")

test_dataset = dataset["test"].cast_column("audio", Audio(sampling_rate=16000))
print(f"Total test samples: {len(test_dataset)}")

# Load Base Model, Processor, and Adapters
print("Loading Base Model and Processor...")
model, processor = FastModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16
)

print(f"Loading LoRA adapters from {CHECKPOINT_DIR}...")
model.load_adapter(CHECKPOINT_DIR)
FastModel.for_inference(model) # Optimize Unsloth for inference

# Inference Loop with Periodic Saving
print("\nStarting Inference...")

inference_buffer = []

for i, sample in enumerate(test_dataset):
    filename = sample.get("filename", f"sample_{i}")
    audio_array = sample["audio"]["array"]
    ground_truth = str(sample.get("transcription", ""))
    lang = sample.get("language", "english").lower()
    
    # Get the correct prompt for the language
    user_prompt = prompts.get(lang, prompts["english"])
    
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
    
    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True, 
        return_tensors='pt'
    ).to(model.device, dtype=torch.bfloat16)
    
    # Generate transcript using beam search
    with torch.no_grad():
        result = model.generate(
            **inputs, 
            max_new_tokens=256, 
            num_beams=6  
        )
        
    # Decode and extract only the model's response
    decoded_result = processor.tokenizer.batch_decode(result, skip_special_tokens=True)
    final_transcription = ''.join(decoded_result[0].split('\nmodel\n')[-1]).strip()
    
    # Store result in buffer
    output_data = {
        "filename": filename,
        "language": lang,
        "ground_truth": ground_truth,
        "predicted_transcription": final_transcription
    }
    inference_buffer.append(output_data)
    
    if (i + 1) % 50 == 0:
        for item in inference_buffer:
            out_path = os.path.join(OUTPUT_DIR, f"{item['filename']}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=4)
        
        print(f"[{i + 1}/{len(test_dataset)}] Processed and saved 50 new transcriptions.")
        inference_buffer.clear() 

# Clean up any remaining items in the buffer
if inference_buffer:
    for item in inference_buffer:
        out_path = os.path.join(OUTPUT_DIR, f"{item['filename']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=4)
    print(f"[{len(test_dataset)}/{len(test_dataset)}] Processed and saved final transcriptions.")

print("\nAll inferences complete! Transcriptions saved to the directory.")
