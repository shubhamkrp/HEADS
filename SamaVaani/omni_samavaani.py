#!/usr/bin/env python3
"""
Fine-tune facebook/omniASR-LLM-3B on a custom multilingual ASR dataset
using LoRA + multi-loss (LM CE + cross-GPU Contrastive + CTC).

Tested with:
    torch==2.6.0  torchaudio==2.6.0  (cu124)
    fairseq2==0.6  fairseq2n==0.6
    omnilingual-asr>=0.2.0  (--no-deps install)
    loralib>=0.1.2

Install (exactly):
    pip install torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
    pip install fairseq2==0.6 fairseq2n==0.6 \
        --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
    pip install "omnilingual-asr>=0.2.0" --no-deps
    pip install loralib datasets jiwer


NOTE: LLaMA decoder uses PREFIX EMBEDDINGS, not cross-attention.
      Audio encoder output is projected and prepended to text token
      embeddings before being processed by the causal decoder.


Usage:
    torchrun --nproc_per_node=4 omni_samavaani.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DISABLE_TORCHCODEC"] = "1"
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
from torch.utils.data import DataLoader
from datasets import load_from_disk, Audio

import omnilingual_asr
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

def _find_seq2seq_batch():
    _candidates = [
        # omnilingual_asr bundles a compatible Seq2SeqBatch used internally
        ("omnilingual_asr.models.wav2vec2_llama.model", "Seq2SeqBatch"),
        ("omnilingual_asr.datasets.task.asr_task",      "Seq2SeqBatch"),
        ("fairseq2.models.seq2seq",                     "Seq2SeqBatch"),
        ("fairseq2.nn.utils.mask",                      "Seq2SeqBatch"),
        ("fairseq2.data.text",                          "Seq2SeqBatch"),
        ("fairseq2.data.text.converters",               "Seq2SeqBatch"),
    ]
    for module_path, cls_name in _candidates:
        try:
            mod = __import__(module_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls
        except (ImportError, ModuleNotFoundError):
            pass

    from dataclasses import dataclass, field
    from typing import Any, Optional

    @dataclass
    class Seq2SeqBatch:
        source_seqs:     "torch.Tensor"
        source_seq_lens: "torch.Tensor"
        target_seqs:     "torch.Tensor"
        target_seq_lens: "torch.Tensor"
        example:         Optional[Any] = None

    return Seq2SeqBatch

Seq2SeqBatch = _find_seq2seq_batch()

import loralib as lora

# "omniASR_LLM_3B_v2" -> v2 checkpoint.
MODEL_CARD      = "omniASR_LLM_3B"
DATASET_DIR     = "asr_dataset"
PROMPT_DIR      = "prompts"
OUTPUT_DIR      = "outputs_omniasr"

# LoRA
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

CTC_BOTTLENECK  = 512

CONTRASTIVE_W   = 0.41
CTC_W           = 0.24

LR              = 5e-5
GRAD_ACCUM      = 8
MAX_STEPS       = 315
WARMUP_STEPS    = 30
MAX_CTC_SEQLEN  = 2048   # cap encoder seq-len before CTC projection
LOGGING_STEPS   = 10
SAVE_STEPS      = 100
PATIENCE        = 10     # early-stop patience (logging intervals)

BATCH_SIZE      = 1      # per GPU; collate doubles it via pitch-shift
SAMPLING_RATE   = 16000
NUM_MEL_BINS    = 80
SEED            = 3407

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch._dynamo.config.cache_size_limit = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model via ASRInferencePipeline
print(f"Loading {MODEL_CARD} via ASRInferencePipeline ...")
pipeline = ASRInferencePipeline(
    model_card=MODEL_CARD,
    device=device,
    dtype=torch.bfloat16,
)
model     = pipeline.model
tokenizer = pipeline.tokenizer   # fairseq2 SentencePiece tokenizer
print("Model loaded successfully.\n")

# print("--- Sub-modules of model ---")
# for name, mod in model.named_children():
#     print(f"  {name}: {type(mod).__name__}")
# print()

# 3. Tokenizer Setup
vocab_info = tokenizer.vocab_info
VOCAB_SIZE = vocab_info.size
PAD_IDX    = vocab_info.pad_idx if vocab_info.pad_idx is not None else 0
BOS_IDX    = vocab_info.bos_idx if vocab_info.bos_idx is not None else 1
EOS_IDX    = vocab_info.eos_idx if vocab_info.eos_idx is not None else 2
token_encoder = tokenizer.create_encoder()   # callable: str -> list[int]
print(f"Vocab={VOCAB_SIZE}  pad={PAD_IDX}  bos={BOS_IDX}  eos={EOS_IDX}\n")

LORA_CANDIDATE_NAMES = {
    "q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "down_proj", "up_proj",
}

def _apply_lora_recursive(parent: nn.Module, r: int, alpha: int, dropout: float) -> None:
    """Depth-first walk: replace qualifying Linear layers with lora.Linear."""
    for attr_name, child in list(parent.named_children()):
        if isinstance(child, nn.Linear) and attr_name in LORA_CANDIDATE_NAMES:
            replacement = lora.Linear(
                child.in_features, child.out_features,
                r=r, lora_alpha=alpha, lora_dropout=dropout,
                merge_weights=False,
            )
            replacement.weight = child.weight     # share pretrained weights
            if child.bias is not None:
                replacement.bias = child.bias
            setattr(parent, attr_name, replacement)
        else:
            _apply_lora_recursive(child, r, alpha, dropout)

# Apply LoRA + Freeze
print("Freezing all parameters ...")
for p in model.parameters():
    p.requires_grad_(False)

if hasattr(model, "decoder"):
    print("Applying LoRA to LLaMA decoder ...")
    _apply_lora_recursive(model.decoder, LORA_R, LORA_ALPHA, LORA_DROPOUT)
    lora.mark_only_lora_as_trainable(model)
else:
    print("WARNING: 'model.decoder' not found. Check the sub-module printout above.")

# Add CTC Head
if hasattr(model, "encoder") and hasattr(model.encoder, "model_dim"):
    ENCODER_DIM = model.encoder.model_dim
elif hasattr(model, "encoder_decoder_proj"):
    ENCODER_DIM = model.encoder_decoder_proj.weight.shape[1]
else:
    ENCODER_DIM = 2048
    print(f"WARNING: could not infer ENCODER_DIM, defaulting to {ENCODER_DIM}")

print(f"Encoder dim = {ENCODER_DIM}")

ctc_head = nn.Sequential(
    nn.Linear(ENCODER_DIM, CTC_BOTTLENECK, bias=False),
    nn.LayerNorm(CTC_BOTTLENECK),
    nn.GELU(),
    nn.Linear(CTC_BOTTLENECK, VOCAB_SIZE, bias=False),
).to(device=device, dtype=torch.bfloat16)

for p in ctc_head.parameters():
    p.requires_grad_(True)

model.add_module("ctc_head", ctc_head)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)\n")

# Load Prompts
LANGUAGES = ["en", "hi", "kn"]
prompts   = {}
for lang in LANGUAGES:
    p = Path(PROMPT_DIR) / f"{lang}.txt"
    prompts[lang] = p.read_text(encoding="utf-8").strip() if p.exists() \
                    else "Please transcribe this audio accurately."

# Dataset
print(f"Loading dataset from '{DATASET_DIR}' ...")
dataset = load_from_disk(DATASET_DIR)
dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

def format_data(example):
    lang = example.get("language", "en")
    example["user_prompt"]   = prompts.get(lang, prompts.get("en", ""))
    example["transcription"] = str(example.get("transcription", "")).strip()
    return example

print("Formatting train split ...")
train_dataset = dataset["train"].map(format_data, batched=False, num_proc=8)

# Audio Feature Extraction
def extract_fbank(audio_np: np.ndarray, sr: int = SAMPLING_RATE) -> torch.Tensor:
    """Raw waveform [T] -> log-Mel filterbank [T', 80] float32."""
    wv = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
    if wv.abs().max() <= 1.0:
        wv = wv * 32768.0    # match fairseq2's waveform_scale 
    return torchaudio.compliance.kaldi.fbank(
        wv,
        num_mel_bins=NUM_MEL_BINS,
        sample_frequency=float(sr),
        use_energy=False,
        dither=0.0,
    )  # [T', 80]

#Collate Function  (pitch-shift augmentation)
def collate_fn(examples):
    src_list     = []
    tgt_list     = []
    ctc_tgt_list = []
    ctc_len_list = []

    for ex in examples:
        audio_np = ex["audio"]["array"]
        sr       = ex["audio"]["sampling_rate"]
        label    = ex["transcription"]

        ids      = list(token_encoder(label))           # raw token ints
        full_seq = [BOS_IDX] + ids + [EOS_IDX]

        ctc_tgt_list.append(torch.tensor(ids, dtype=torch.long))
        ctc_len_list.append(len(ids))

        # Original
        src_list.append(extract_fbank(audio_np, sr))
        tgt_list.append(torch.tensor(full_seq, dtype=torch.long))

        # Pitch-shifted copy
        wv = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        shifted = F_audio.pitch_shift(wv, sr, random.choice([-5, 5]))
        src_list.append(extract_fbank(shifted.squeeze(0).numpy(), sr))
        tgt_list.append(torch.tensor(full_seq, dtype=torch.long))
        ctc_tgt_list.append(torch.tensor(ids, dtype=torch.long))
        ctc_len_list.append(len(ids))

    B = len(src_list)

    # Pad source
    max_src  = max(f.shape[0] for f in src_list)
    src_seqs = torch.zeros(B, max_src, NUM_MEL_BINS, dtype=torch.float32)
    src_lens = torch.zeros(B, dtype=torch.long)
    for i, f in enumerate(src_list):
        src_seqs[i, :f.shape[0]] = f
        src_lens[i] = f.shape[0]

    # Pad target
    max_tgt  = max(t.shape[0] for t in tgt_list)
    tgt_seqs = torch.full((B, max_tgt), PAD_IDX, dtype=torch.long)
    tgt_lens = torch.zeros(B, dtype=torch.long)
    for i, t in enumerate(tgt_list):
        tgt_seqs[i, :t.shape[0]] = t
        tgt_lens[i] = t.shape[0]

    # LM labels: strip BOS, mask pads with -100
    tgt_labels = tgt_seqs[:, 1:].clone()
    tgt_labels[tgt_labels == PAD_IDX] = -100

    ctc_targets     = torch.cat(ctc_tgt_list)
    ctc_target_lens = torch.tensor(ctc_len_list, dtype=torch.long)

    return dict(
        src_seqs=src_seqs,
        src_lens=src_lens,
        tgt_seqs=tgt_seqs,
        tgt_lens=tgt_lens,
        tgt_labels=tgt_labels,
        ctc_targets=ctc_targets,
        ctc_target_lens=ctc_target_lens,
    )

_enc_cache: dict = {}

def _encoder_forward_hook(module, inp, output):
    """Store encoder output after each encoder forward call."""
    if isinstance(output, tuple):
        _enc_cache["hidden"] = output[0]
    elif torch.is_tensor(output):
        _enc_cache["hidden"] = output
    elif hasattr(output, "seqs"):           # fairseq2 SequenceBatch output
        _enc_cache["hidden"] = output.seqs
    else:
        # Fallback: try first positional element
        try:
            _enc_cache["hidden"] = output[0]
        except Exception:
            _enc_cache["hidden"] = None

_hook_handle = model.encoder.register_forward_hook(_encoder_forward_hook)

ctc_loss_fn = nn.CTCLoss(blank=PAD_IDX, zero_infinity=True)


def compute_multi_loss(model, batch):
    _enc_cache.clear()

    src_seqs   = batch["src_seqs"].to(device, dtype=torch.bfloat16)
    src_lens   = batch["src_lens"].to(device)
    tgt_seqs   = batch["tgt_seqs"].to(device)
    tgt_lens   = batch["tgt_lens"].to(device)
    tgt_labels = batch["tgt_labels"].to(device)
    ctc_tgts   = batch["ctc_targets"].to(device)
    ctc_tlens  = batch["ctc_target_lens"].to(device)
    B          = src_seqs.shape[0]

    seq_batch = Seq2SeqBatch(
        source_seqs     = src_seqs,
        source_seq_lens = src_lens,
        target_seqs     = tgt_seqs,
        target_seq_lens = tgt_lens,
    )

    # Full forward; encoder hook fires inside and fills _enc_cache
    # model(Seq2SeqBatch) returns LM logits [B, T_text, VOCAB_SIZE]
    lm_logits = model(seq_batch)

    lm_loss = F.cross_entropy(
        lm_logits.reshape(-1, VOCAB_SIZE).float(),
        tgt_labels.reshape(-1),
        ignore_index=-100,
    )

    enc_hidden = _enc_cache.get("hidden")
    if enc_hidden is None:
        raise RuntimeError(
            "Encoder forward hook did not capture hidden states. "
            "Verify that model.encoder is a direct nn.Module child and "
            "that the hook handle has not been removed."
        )

    # Cap encoder sequence length before expensive CTC projection
    actual_max = min(int(src_lens.max().item()), MAX_CTC_SEQLEN)
    enc_sliced = enc_hidden[:, :actual_max, :].contiguous()   # [B, T', D]
    del enc_hidden
    torch.cuda.empty_cache()

    #Contrastive Loss
    pooled = enc_sliced.float().mean(dim=1)    # [B, D]
    pooled = F.normalize(pooled, p=2, dim=1)

    if model.training and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size()
        gathered = [torch.zeros_like(pooled) for _ in range(ws)]
        torch.distributed.all_gather(gathered, pooled)
        global_rep = torch.cat(gathered, dim=0)
        gB = global_rep.shape[0]
        if gB > 1 and gB % 2 == 0:
            sim = torch.matmul(global_rep, global_rep.T) / 0.05
            clabels = torch.empty(gB, dtype=torch.long, device=device)
            for i in range(0, gB, 2):
                clabels[i]     = i + 1
                clabels[i + 1] = i
            sim.masked_fill_(torch.eye(gB, device=device).bool(), -9e15)
            contrastive_loss = F.cross_entropy(sim, clabels) / ws
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
    else:
        contrastive_loss = torch.tensor(0.0, device=device)

    #  CTC Loss 
    torch.cuda.empty_cache()
    unwrapped  = model.module if hasattr(model, "module") else model
    ctc_dtype  = next(unwrapped.ctc_head.parameters()).dtype
    ctc_logits = unwrapped.ctc_head(enc_sliced.to(ctc_dtype))              # [B, T', V]
    log_probs  = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1) # [T', B, V]

    ctc_in_lens = src_lens.clamp(max=actual_max).long()
    ctc_tl_safe = torch.minimum(ctc_tlens, ctc_in_lens)

    if ctc_tl_safe.sum() > 0:
        ctc_loss = ctc_loss_fn(log_probs, ctc_tgts, ctc_in_lens, ctc_tl_safe)
    else:
        ctc_loss = torch.tensor(0.0, device=device)

    total = lm_loss + CONTRASTIVE_W * contrastive_loss + CTC_W * ctc_loss
    return total, lm_loss, contrastive_loss, ctc_loss


# Training-Loss Early Stopping
class TrainingLossEarlyStop:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best        = float("inf")
        self.counter     = 0
        self.should_stop = False

    def __call__(self, step: int, loss: float) -> bool:
        if loss < self.best - self.min_delta:
            self.best    = loss
            self.counter = 0
            print(f"  => [Step {step}] loss improved to {self.best:.4f}")
        else:
            self.counter += 1
            print(f"  => [Step {step}] no improvement. "
                  f"patience={self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("  => Early stopping triggered.")
                self.should_stop = True
        return self.should_stop

# Optimiser & LR Schedule  (cosine with linear warmup)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01, eps=1e-8)

def _lr_lambda(step: int) -> float:
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    t = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * t)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    prefetch_factor=2,
    pin_memory=True,
)

# Training Loop
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
model.train()
optimizer.zero_grad()

early_stop  = TrainingLossEarlyStop(patience=PATIENCE)
global_step = 0
accum_loss  = 0.0
start_time  = time.time()

print("Starting training ...\n")

DONE = False
for epoch in range(9999):
    if DONE:
        break
    for batch in train_loader:
        if global_step >= MAX_STEPS:
            DONE = True
            break

        total_loss, lm_l, cont_l, ctc_l = compute_multi_loss(model, batch)
        (total_loss / GRAD_ACCUM).backward()
        accum_loss += total_loss.item()

        if (global_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % LOGGING_STEPS == 0:
            avg    = accum_loss / LOGGING_STEPS
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[{global_step:>4}]  total={avg:.4f}  lm={lm_l.item():.4f}"
                  f"  cont={cont_l.item():.4f}  ctc={ctc_l.item():.4f}"
                  f"  lr={lr_now:.2e}")
            if early_stop(global_step, avg):
                DONE = True
                break
            accum_loss = 0.0

        if global_step % SAVE_STEPS == 0:
            ckpt = Path(OUTPUT_DIR) / f"step_{global_step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save(lora.lora_state_dict(model), ckpt / "lora_weights.pt")
            torch.save(model.ctc_head.state_dict(),  ckpt / "ctc_head.pt")
            print(f"  => Checkpoint saved -> {ckpt}")

elapsed = time.time() - start_time
print(f"\n--- Training complete ---")
print(f"Total: {elapsed/60:.1f} min  ({elapsed/3600:.2f} hr)")



final_dir = Path(OUTPUT_DIR) / "final_model"
final_dir.mkdir(parents=True, exist_ok=True)
torch.save(lora.lora_state_dict(model), final_dir / "lora_weights.pt")
torch.save(model.ctc_head.state_dict(),  final_dir / "ctc_head.pt")
print(f"\nSaved LoRA adapters + CTC head -> {final_dir}")

# Clean up the encoder hook
_hook_handle.remove()
print("Done.")
