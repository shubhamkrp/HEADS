import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as F_audio
import numpy as np
import jiwer
from unsloth import FastModel
from datasets import load_from_disk, Audio
from transformers import AutoProcessor, TrainerCallback
from evaluate import load
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

# Add Trainable Low-Rank CTC Head 
vocab_size = len(processor.tokenizer)
hidden_size = model.config.text_config.hidden_size
bottleneck_dim = 512  # low-rank bottleneck size

# create a sequential bottleneck to drastically reduce parameters
ctc_head_module = nn.Sequential(
    nn.Linear(hidden_size, bottleneck_dim, bias=False),
    nn.LayerNorm(bottleneck_dim), # stabilize training with bottlenecks
    nn.GELU(),
    nn.Linear(bottleneck_dim, vocab_size, bias=False)
).to(model.device, dtype=torch.bfloat16)

model.add_module("ctc_head", ctc_head_module)

# ensure the new head requires gradients
for param in model.ctc_head.parameters():
    param.requires_grad = True

# load Prompts per Language
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

# load and Prepare the Custom Dataset
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
        
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        n_steps = random.choice([-5, 5]) 
        shifted_waveform = F_audio.pitch_shift(waveform, sr, n_steps)
        
        texts.append(text)
        audios.append(shifted_waveform.squeeze(0).numpy())
        
    batch = processor(text=texts, audio=audios, return_tensors='pt', padding=True)
    
    labels = batch['input_ids'].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    for attr in ['image_token_id', 'audio_token_id', 'boi_token_id', 'eoi_token_id']:
        if hasattr(processor.tokenizer, attr):
            token_id = getattr(processor.tokenizer, attr)
            labels[labels == token_id] = -100
            
    batch['labels'] = labels
    return batch

# Custom Trainer for LM & CTC Loss
class CustomMultiLossSFTTrainer(SFTTrainer):
    def __init__(self, lm_weight=0.5, ctc_weight=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_weight = lm_weight
        self.ctc_weight = ctc_weight
        
        blank_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
        self.ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs, output_hidden_states=True)
        lm_loss = outputs.loss
        
        hidden_states = outputs.hidden_states[-1] 
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        
        true_batch_size = labels.shape[0]
        seq_len = labels.shape[1]
        
        # Ensure correct tensor dimensions
        if hidden_states.shape[0] != true_batch_size:
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.reshape(true_batch_size, seq_len, -1)
            elif hidden_states.shape[1] == true_batch_size:
                hidden_states = hidden_states.transpose(0, 1)
                
        batch_size = hidden_states.shape[0] 
        hidden_dim = hidden_states.shape[-1]
        hidden_states_2d = hidden_states.reshape(batch_size, -1, hidden_dim)

        MAX_CTC_SEQ_LEN = 2048
        if attention_mask is not None:
            actual_max_len = attention_mask.sum(dim=1).max().long().item()
            actual_max_len = min(actual_max_len, seq_len, MAX_CTC_SEQ_LEN)
        else:
            actual_max_len = min(seq_len, MAX_CTC_SEQ_LEN)
            
        hidden_states_sliced = hidden_states_2d[:, :actual_max_len, :]
        
        del hidden_states
        del hidden_states_2d
        torch.cuda.empty_cache()

        # CTC Loss Calculation
        unwrapped_model = model.module if hasattr(model, "module") else model
        
        ctc_dtype = next(unwrapped_model.ctc_head.parameters()).dtype
        ctc_logits = unwrapped_model.ctc_head(hidden_states_sliced.to(ctc_dtype))
        
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        
        target_lengths = []
        targets_list = []
        
        for i in range(batch_size):
            valid_labels = labels[i, :actual_max_len]
            valid_labels = valid_labels[valid_labels != -100]
            
            if len(valid_labels) > actual_max_len:
                valid_labels = valid_labels[:actual_max_len]
                
            target_lengths.append(len(valid_labels))
            targets_list.append(valid_labels)
            
        if sum(target_lengths) > 0:
            targets = torch.cat(targets_list)
            
            if attention_mask is not None:
                input_lengths = torch.clamp(attention_mask.sum(dim=-1).long(), max=actual_max_len)
            else:
                input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long, device=log_probs.device)
                
            target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long, device=log_probs.device)
            
            # Cast to float32 only inside the loss function for numerical stability
            ctc_loss = self.ctc_loss_fn(log_probs.float(), targets, input_lengths, target_lengths_tensor)
        else:
            ctc_loss = torch.tensor(0.0, device=log_probs.device)
            
        print(f"LM Loss: {lm_loss.item():.4f} | CTC Loss: {ctc_loss.item():.4f}")

        total_loss = (self.lm_weight * lm_loss) + (self.ctc_weight * ctc_loss)

        return (total_loss, outputs) if return_outputs else total_loss
    
# Custom Training Loss Early Stopping
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

trainer = CustomMultiLossSFTTrainer(
    lm_weight=0.5,
    ctc_weight=0.5, 
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
        output_dir='outputs-cl',
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

print("Saving final model adapters and CTC head...")
final_save_path = "outputs-cl/final_model"
trainer.save_model(final_save_path)
processor.save_pretrained(final_save_path)

torch.save(model.ctc_head.state_dict(), os.path.join(final_save_path, "ctc_head.pt"))
print("All artifacts saved successfully.")