import os
import random
import torch
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

# Load Model and Processor (E4B Version)
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
    r=16,
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
    """Format dataset to match expected Gemma 3n message format."""
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
train_dataset = dataset["train"].map(format_data, batched=False)
dev_dataset = dataset["dev"].map(format_data, batched=False)

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
        
        # Append Augmented Audio (Pitch shift for gender fairness: +/- 5 semitones)
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

# Custom Trainer for Contrastive Loss
class ContrastiveSFTTrainer(SFTTrainer):
    def __init__(self, contrastive_weight=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs, output_hidden_states=True)
        lm_loss = outputs.loss
        
        # Extract the final layer's hidden states
        hidden_states = outputs.hidden_states[-1] 
        
        # Guarantee 2D pooling by explicitly flattening any extra multimodal/chunk dimensions
        batch_size = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[-1]
        
        # Reshape to strictly (batch_size, total_sequence_length, hidden_dim)
        hidden_states_2d = hidden_states.view(batch_size, -1, hidden_dim)
        
        # Mean-pool across the sequence length to get exactly (batch_size, hidden_dim)
        pooled_rep = hidden_states_2d.mean(dim=1)
        pooled_rep = F.normalize(pooled_rep, p=2, dim=1)
        
        # Now pooled_rep is 2D, so pooled_rep.T safely transposes to (hidden_dim, batch_size)
        if batch_size % 2 == 0:
            temperature = 0.05
            sim_matrix = torch.matmul(pooled_rep, pooled_rep.T) / temperature
            
            # Construct targets to pull consecutive pairs (Original, Augmented) together
            labels = torch.empty(batch_size, dtype=torch.long, device=pooled_rep.device)
            for i in range(0, batch_size, 2):
                labels[i] = i + 1
                labels[i+1] = i
                
            # Mask self-similarity (diagonal)
            mask = torch.eye(batch_size, device=sim_matrix.device).bool()
            sim_matrix.masked_fill_(mask, -9e15)
            
            contrastive_loss = F.cross_entropy(sim_matrix, labels)
            total_loss = lm_loss + (self.contrastive_weight * contrastive_loss)
        else:
            total_loss = lm_loss

        return (total_loss, outputs) if return_outputs else total_loss

# Early Stopping Callback
class MedianWEREarlyStoppingCallback(TrainerCallback):
    def __init__(self, dev_dataset, processor, model, patience=3):
        self.dev_dataset = dev_dataset
        self.processor = processor
        self.model = model
        self.patience = patience
        self.best_wer = float('inf')
        self.patience_counter = 0
        self.wer_metric = load('wer')
        self.text_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
        ])
        
    def on_evaluate(self, args, state, control, **kwargs):
        print("\n--- Evaluating Median WER on Dev Set ---")
        self.model.eval()
        wers = []
        
        for sample in self.dev_dataset:
            lang = sample.get('language', 'en')
            user_prompt = prompts.get(lang, prompts.get('en'))
            
            messages = [
                {'role': 'system', 'content': [{'type': 'text', 'text': 'You are an assistant that transcribes speech accurately.'}]},
                {'role': 'user', 'content': [{'type': 'audio', 'audio': sample['audio']['array']}, {'type': 'text', 'text': user_prompt}]}
            ]
            
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt'
            ).to('cuda', dtype=torch.bfloat16)
            
            with torch.no_grad():
                result = self.model.generate(**inputs, max_new_tokens=256, num_beams=6)
                
            decoded_result = self.processor.tokenizer.batch_decode(result, skip_special_tokens=True)
            final_result = ''.join(decoded_result[0].split('\nmodel\n')[-1])
            
            ref = self.text_transform(sample['transcription'])
            pred = self.text_transform(final_result)
            
            try:
                if ref and pred: 
                    wers.append(self.wer_metric.compute(references=[ref], predictions=[pred]))
            except Exception:
                pass
                
        median_wer = np.median(wers) * 100 if wers else float('inf')
        print(f"Median WER on Dev Set: {median_wer:.2f}%")
        
        if median_wer < self.best_wer:
            self.best_wer = median_wer
            self.patience_counter = 0
            print("New best median WER! Model is improving.")
        else:
            self.patience_counter += 1
            print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                print("Early stopping triggered due to lack of WER improvement!")
                control.should_training_stop = True

        self.model.train()

# Initialization and Training
early_stopping_callback = MedianWEREarlyStoppingCallback(dev_dataset, processor, model, patience=3)

trainer = ContrastiveSFTTrainer(
    contrastive_weight=0.1,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=processor.tokenizer,
    data_collator=collate_fn_with_augmentation,
    callbacks=[early_stopping_callback],
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        optim='adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        seed=3407,
        output_dir='outputs',
        report_to='none',
        remove_unused_columns=False,
        dataset_text_field='',
        dataset_kwargs={'skip_prepare_dataset': True},
        dataset_num_proc=4,
        max_length=2048,
    )
)

print("Starting training...")
trainer.train()

model.save_pretrained('gemma-3n-E4B-contrastive-finetuned')
processor.save_pretrained('gemma-3n-E4B-contrastive-finetuned')
print("Model saved successfully.")
