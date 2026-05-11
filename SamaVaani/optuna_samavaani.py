import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import random
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as F_audio
import numpy as np
import optuna
from unsloth import FastModel
from datasets import load_from_disk, Audio
from transformers import AutoProcessor
from trl import SFTTrainer, SFTConfig

torch._dynamo.config.cache_size_limit = 32

# Global Setup & Prompts
model_name = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
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
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def format_data(example):
    lang = example.get('language', 'en')
    user_prompt = prompts.get(lang, prompts.get('en'))
    
    audio = example['audio']['array']
    label = str(example['transcription'])
    
    message = [
        {'role': 'system', 'content': [{'type': 'text', 'text': 'You are an assistant that transcribes speech accurately.'}]},
        {'role': 'user', 'content': [{'type': 'audio', 'audio': audio}, {'type': 'text', 'text': user_prompt}]},
        {'role': 'model', 'content': [{'type': 'text', 'text': label}]}
    ]
    
    example['messages'] = message
    return example

print("Formatting splits...")
train_dataset = dataset["train"].map(format_data, batched=False, num_proc=8)
dev_dataset = dataset["dev"].map(format_data, batched=False, num_proc=8)

# Collate Function
_, global_processor = FastModel.from_pretrained(model_name=model_name, max_seq_length=1024, load_in_4bit=True)
global_vocab_size = len(global_processor.tokenizer)

def collate_fn_with_augmentation(examples):
    texts = []
    audios = []
    
    for example in examples:
        text = global_processor.apply_chat_template(
            example['messages'], tokenize=False, add_generation_prompt=False
        ).strip()
        
        audio_array = example['audio']['array']
        sr = example['audio']['sampling_rate']
        
        # Pitch shift augmentation
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        n_steps = random.choice([-5, 5]) 
        shifted_waveform = F_audio.pitch_shift(waveform, sr, n_steps)
        
        texts.append(text)
        audios.append(shifted_waveform.squeeze(0).numpy())
        
    batch = global_processor(text=texts, audio=audios, return_tensors='pt', padding=True)
    
    labels = batch['input_ids'].clone()
    labels[labels == global_processor.tokenizer.pad_token_id] = -100
    
    for attr in ['image_token_id', 'audio_token_id', 'boi_token_id', 'eoi_token_id']:
        if hasattr(global_processor.tokenizer, attr):
            token_id = getattr(global_processor.tokenizer, attr)
            labels[labels == token_id] = -100
            
    batch['labels'] = labels
    return batch

del _ 
torch.cuda.empty_cache()

# Custom Trainer
class CustomMultiLossSFTTrainer(SFTTrainer):
    def __init__(self, lm_weight=0.5, contrastive_weight=0.2, ctc_weight=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_weight = lm_weight
        self.contrastive_weight = contrastive_weight
        self.ctc_weight = ctc_weight
        
        blank_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
        # Prevent zero_infinity from masking an underlying issue during debugging
        self.ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs, output_hidden_states=True)
        lm_loss = outputs.loss
        
        hidden_states = outputs.hidden_states[-1] 
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        
        true_batch_size = labels.shape[0]
        seq_len = labels.shape[1]
        
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
        del hidden_states, hidden_states_2d
        
        # Contrastive Loss
        pooled_rep = hidden_states_sliced.mean(dim=1)
        pooled_rep = F.normalize(pooled_rep, p=2, dim=1)
        
        if self.model.training and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_reps = [torch.zeros_like(pooled_rep) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_reps, pooled_rep)
            global_pooled_rep = torch.cat(gathered_reps, dim=0)
            global_batch_size = global_pooled_rep.shape[0]
            
            if global_batch_size > 1 and global_batch_size % 2 == 0:
                temperature = 0.05
                sim_matrix = torch.matmul(global_pooled_rep, global_pooled_rep.T) / temperature
                contrastive_labels = torch.empty(global_batch_size, dtype=torch.long, device=pooled_rep.device)
                for i in range(0, global_batch_size, 2):
                    contrastive_labels[i] = i + 1
                    contrastive_labels[i+1] = i
                    
                mask = torch.eye(global_batch_size, device=sim_matrix.device).bool()
                sim_matrix.masked_fill_(mask, -9e15)
                contrastive_loss = F.cross_entropy(sim_matrix, contrastive_labels) / world_size
            else:
                contrastive_loss = torch.tensor(0.0, device=pooled_rep.device)
        else:
            contrastive_loss = torch.tensor(0.0, device=pooled_rep.device)

        # CTC Loss
        unwrapped_model = model.module if hasattr(model, "module") else model
        ctc_dtype = next(unwrapped_model.ctc_head.parameters()).dtype
        ctc_logits = unwrapped_model.ctc_head(hidden_states_sliced.to(ctc_dtype))
        
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        target_lengths = []
        targets_list = []
        
        for i in range(batch_size):
            valid_labels = labels[i, :actual_max_len]
            # Strip out padding
            valid_labels = valid_labels[valid_labels != -100]
            
            # Ensure valid bounds for tokens to prevent index assertions
            valid_labels = torch.clamp(valid_labels, min=0, max=global_vocab_size - 1)

            # Truncate text targets to fit within the physical limit of the input projection
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
            
            input_lengths = torch.max(input_lengths, target_lengths_tensor)

            ctc_loss = self.ctc_loss_fn(log_probs.float(), targets, input_lengths, target_lengths_tensor)
        else:
            ctc_loss = torch.tensor(0.0, device=log_probs.device)

        # Apply normalized multipliers here
        total_loss = (self.lm_weight * lm_loss) + (self.contrastive_weight * contrastive_loss) + (self.ctc_weight * ctc_loss)

        return (total_loss, outputs) if return_outputs else total_loss

# Optuna Objective Definition
best_val_loss = float('inf')
best_save_path = "outputs_optuna/best_tuned_model"

def objective(trial):
    global best_val_loss
    
    # Suggest raw multipliers
    raw_lm = trial.suggest_float("raw_lm", 0.1, 0.8)
    raw_con = trial.suggest_float("raw_con", 0.1, 0.8)
    raw_ctc = 1-raw_lm-raw_con
    
    total_weight = raw_lm + raw_con + raw_ctc
    lm_weight = raw_lm / total_weight
    contrastive_weight = raw_con / total_weight
    ctc_weight = raw_ctc / total_weight
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Normalized Weights -> LM: {lm_weight:.4f} | Con: {contrastive_weight:.4f} | CTC: {ctc_weight:.4f}")

    # Re-initialize the model to start fresh for each trial
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

    vocab_size = len(processor.tokenizer)
    hidden_size = model.config.text_config.hidden_size
    bottleneck_dim = 512

    ctc_head_module = nn.Sequential(
        nn.Linear(hidden_size, bottleneck_dim, bias=False),
        nn.LayerNorm(bottleneck_dim),
        nn.GELU(),
        nn.Linear(bottleneck_dim, vocab_size, bias=False)
    ).to(model.device, dtype=torch.bfloat16)

    model.add_module("ctc_head", ctc_head_module)
    for param in model.ctc_head.parameters():
        param.requires_grad = True

    trainer = CustomMultiLossSFTTrainer(
        lm_weight=lm_weight,
        contrastive_weight=contrastive_weight,
        ctc_weight=ctc_weight,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset, # Evaluate on Validation
        processing_class=processor.tokenizer,
        data_collator=collate_fn_with_augmentation,
        args=SFTConfig(
            per_device_train_batch_size=1,       
            gradient_accumulation_steps=8,       
            max_length=1024,                     
            dataloader_num_workers=4, 
            dataloader_prefetch_factor=2,
            ddp_find_unused_parameters=False,
            bf16=True,
            warmup_ratio=0.1,
            learning_rate=5e-5,
            logging_steps=10,
            eval_strategy="steps",     
            eval_steps=100,            
            save_strategy='no',        
            optim='adamw_8bit',
            weight_decay=0.01,
            lr_scheduler_type='cosine',
            seed=3407,
            output_dir=f'outputs_optuna/trial_{trial.number}',
            report_to='none',
            remove_unused_columns=False,
            dataset_kwargs={'skip_prepare_dataset': True},
        )
    )

    # Train for 100 steps
    trainer.train()

    # Evaluate on dev set
    metrics = trainer.evaluate()
    val_loss = metrics.get("eval_loss", float('inf'))
    print(f"Trial {trial.number} finished with Eval Loss: {val_loss:.4f}")

    # Save logic if it's the best model so far
    if val_loss < best_val_loss:
        print(f"--> New Best Model Found! Val Loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving...")
        best_val_loss = val_loss
        
        os.makedirs(best_save_path, exist_ok=True)
        trainer.save_model(best_save_path)
        processor.save_pretrained(best_save_path)
        torch.save(model.ctc_head.state_dict(), os.path.join(best_save_path, "ctc_head.pt"))

    # Cleanup aggressively to prevent Memory Leaks across trials
    del trainer, model, processor, ctc_head_module
    gc.collect()
    torch.cuda.empty_cache()

    return val_loss

# Execute Optuna Tuning
if __name__ == "__main__":
    print("Initializing Optuna Study...")
    study = optuna.create_study(direction="minimize", study_name="loss_weight_tuning")
    
    n_trials = 20 
    study.optimize(objective, n_trials=n_trials)

    print("\n=============================================")
    print("Tuning Complete!")
    print(f"Best Eval Loss: {study.best_value:.4f}")
    
    # Retrieve the best raw params and re-normalize them for the final printout
    best_raw_lm = study.best_params["raw_lm"]
    best_raw_con = study.best_params["raw_con"]
    best_raw_ctc = study.best_params["raw_ctc"]
    
    best_total = best_raw_lm + best_raw_con + best_raw_ctc
    print("Best Multipliers (Sum = 1):")
    print(f"  LM Weight:          {best_raw_lm / best_total:.4f}")
    print(f"  Contrastive Weight: {best_raw_con / best_total:.4f}")
    print(f"  CTC Weight:         {best_raw_ctc / best_total:.4f}")
    print(f"Best model saved successfully at: {best_save_path}")
    print("=============================================")
