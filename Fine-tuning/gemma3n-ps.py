import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import random
import torch
import torchaudio.functional as F_audio
import numpy as np
from unsloth import FastModel
from datasets import load_from_disk, Audio
from transformers import AutoProcessor, TrainerCallback
from trl import SFTTrainer, SFTConfig

torch._dynamo.config.cache_size_limit = 32

# Load Model and Processor
model_name = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"

model, processor = FastModel.from_pretrained(
    model_name=model_name,
    dtype=None,
    max_seq_length=1024,
    load_in_4bit=True,
    full_finetuning=False
)

model = FastModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load Prompts per Language
languages = ["en", "hi", "kn"]
prompts = {}

print("Loading prompts...")
for lang in languages:
    prompt_path = os.path.join("prompts", f"{lang}.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts[lang] = f.read().strip()
    else:
        prompts[lang] = "Please transcribe this audio."

# Load and Prepare the Custom Dataset
print("Loading dataset from disk...")
dataset = load_from_disk("asr_dataset")

# Standardize the audio inputs to mono, 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def format_data(example):
    lang = example.get('language', 'en')
    user_prompt = prompts.get(lang, prompts.get('en'))
    
    audio = example['audio']['array']
    label = str(example['transcription'])
    
    message = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': 'You are an assistant that transcribes speech accurately.'}]
        },
        {
            'role': 'user',
            'content': [
                {'type': 'audio', 'audio': audio},
                {'type': 'text', 'text': user_prompt}
            ]
        },
        {
            'role': 'model',
            'content': [{'type': 'text', 'text': label}]
        }
    ]
    
    example['messages'] = message
    return example

print("Formatting splits...")
train_dataset = dataset["train"].map(format_data, batched=False, num_proc=8)

# Collate Function with On-the-Fly Pitch Shift
def collate_fn_with_augmentation(examples):
    texts = []
    audios = []
    
    for example in examples:
        text = processor.apply_chat_template(
            example['messages'], tokenize=False, add_generation_prompt=False
        ).strip()
        
        audio_array = example['audio']['array']
        sr = example['audio']['sampling_rate']
        
        texts.append(text)
        audios.append(audio_array)
        
        # Pitch-Shifted Audio
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        n_steps = random.choice([-5, 5]) 
        shifted_waveform = F_audio.pitch_shift(waveform, sr, n_steps)
        
        texts.append(text)
        audios.append(shifted_waveform.squeeze(0).numpy())
        
    batch = processor(text=texts, audio=audios, return_tensors='pt', padding=True)
    
    # Setup labels for standard LM loss
    labels = batch['input_ids'].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    for attr in ['image_token_id', 'audio_token_id', 'boi_token_id', 'eoi_token_id']:
        if hasattr(processor.tokenizer, attr):
            token_id = getattr(processor.tokenizer, attr)
            if token_id is not None:
                labels[labels == token_id] = -100
            
    batch['labels'] = labels
    return batch
    
# Training Loss Early Stopping
class TrainingLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_loss = float('inf')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            current_loss = logs["loss"]
            
            if current_loss < (self.best_loss - self.min_delta):
                self.best_loss = current_loss
                self.patience_counter = 0
                print(f"\n=> [Step {state.global_step}] Training loss improved to {self.best_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"\n=> [Step {state.global_step}] Training loss did not improve. Patience: {self.patience_counter}/{self.patience}")
                
                if self.patience_counter >= self.patience:
                    print(f"=> Early stopping triggered: Training loss hasn't improved for {self.patience} logging steps.")
                    control.should_training_stop = True

# Initialization and Training
train_loss_early_stopping = TrainingLossEarlyStoppingCallback(patience=10)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    processing_class=processor.tokenizer,
    data_collator=collate_fn_with_augmentation,
    callbacks=[train_loss_early_stopping],
    args=SFTConfig(
        per_device_train_batch_size=1,       
        gradient_accumulation_steps=8,       
        max_length=1024,                     
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
        ddp_find_unused_parameters=False,
        bf16=True,
        warmup_ratio=0.1,
        num_train_epochs=2,
        learning_rate=5e-5,
        logging_steps=10,
        eval_strategy="no",        
        save_strategy='steps',
        save_steps=100,
        optim='adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        seed=3407,
        output_dir='outputs-ps',
        report_to='none',
        remove_unused_columns=False,
        dataset_text_field='',
        dataset_kwargs={'skip_prepare_dataset': True},
    )
)

print("Starting training...")
start_time = time.time()

trainer.train()

end_time = time.time()
training_duration = end_time - start_time

print(f"\n--- Training Complete ---")
print(f"Total time required for fine-tuning: {training_duration / 60:.2f} minutes ({training_duration / 3600:.2f} hours)")


print("Saving final model adapters...")
final_save_path = "outputs-ps/final_model"
trainer.save_model(final_save_path)
processor.save_pretrained(final_save_path)

print("All artifacts saved successfully.")