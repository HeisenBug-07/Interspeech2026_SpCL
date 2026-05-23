import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Project-specific imports
import dataloader as ds
import model_w2v2_roberta as ml 

def get_args():
    parser = argparse.ArgumentParser(description="Dual Encoder Finetuning (Wav2Vec2 + RoBERTa)")
    parser.add_argument("--train_txt", type=str, default="train_ekstep_md.txt")
    parser.add_argument("--val_txt", type=str, default="val_ekstep_md.txt")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt_wav2vec_roberta_0.5_seq") 
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--lr", type=float, default=5e-5)     
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_lang", type=int, default=12)
    parser.add_argument("--max_wav_len", type=int, default=160000) 
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--contrast_weight", type=float, default=0.3)
    parser.add_argument("--class_weight", type=float, default=0.7)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # --- 1. INITIALIZE LOG FILE ---
    log_path = os.path.join(args.ckpt_dir, "log.txt")
    # Write the header
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,contrast_acc,class_acc,balanced_acc\n")

    model = ml.build_model(args.num_lang)

    # --- FREEZING STRATEGY ---
    for p in model.parameters(): p.requires_grad = False
    
    # Unfreeze Wav2Vec2/RoBERTa layers (top 4)
    for layer in model.audio_branch.wav2vec2.encoder.layers[-4:]:
        for p in layer.parameters(): p.requires_grad = True
    for layer in model.text_branch.roberta.encoder.layer[-4:]:
        for p in layer.parameters(): p.requires_grad = True
            
    if hasattr(model.text_branch.roberta, "pooler") and model.text_branch.roberta.pooler:
        for p in model.text_branch.roberta.pooler.parameters(): p.requires_grad = True

    # Unfreeze Projections/Heads
    for p in model.audio_branch.projection.parameters(): p.requires_grad = True
    for p in model.text_branch.projection.parameters(): p.requires_grad = True
    for p in model.lang_classifier.parameters(): p.requires_grad = True
    model.logit_scale.requires_grad = True

    # Optimizer & Data
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience)

    ce_weights = ds.compute_class_weights(args.train_txt, args.num_lang) if args.use_class_weights else None
    if ce_weights is not None: ce_weights = ce_weights.to(ml.DEVICE)
    criterion_ce = nn.CrossEntropyLoss(weight=ce_weights)

    train_ds = ds.LIDDataset(args.train_txt, sr=args.sr)
    val_ds = ds.LIDDataset(args.val_txt, sr=args.sr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ds.collate_fn, num_workers=4)

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss, train_batches = 0.0, 0

        for wavs, caps, langs, doms in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            wavs, langs, doms = wavs.to(ml.DEVICE), langs.to(ml.DEVICE), doms.to(ml.DEVICE)
            optimizer.zero_grad()

            audio_emb = ml.encode_audio(model, wavs, args.max_wav_len)
            text_emb = ml.encode_text_prompts(model, caps)
            logits = ml.compute_similarity_logits(audio_emb, text_emb, model)
            T = ml.build_target_matrix(langs, doms, device=ml.DEVICE)

            loss_contrast = 0.5 * (ml.weighted_contrastive_loss(logits, T) + ml.weighted_contrastive_loss(logits.T, T.T))
            loss_class = criterion_ce(model.lang_classifier(audio_emb), langs)
            
            loss = (args.contrast_weight * loss_contrast) + (args.class_weight * loss_class)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1
            if train_batches % 100 == 0:
                with torch.no_grad(): model.logit_scale.data.clamp_(max=np.log(100))

        # --- Validation Loop ---
        model.eval()
        contrast_correct, contrast_total, val_loss_sum, val_batches = 0, 0, 0.0, 0
        y_true_cls, y_pred_cls = [], []

        with torch.no_grad():
            for wavs, caps, langs, doms in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                wavs, langs, doms = wavs.to(ml.DEVICE), langs.to(ml.DEVICE), doms.to(ml.DEVICE)
                
                audio_emb = ml.encode_audio(model, wavs, args.max_wav_len)
                text_emb = ml.encode_text_prompts(model, caps)
                logits = ml.compute_similarity_logits(audio_emb, text_emb, model)
                
                T_val = ml.build_target_matrix(langs, doms, device=ml.DEVICE)
                v_loss_cont = 0.5 * (ml.weighted_contrastive_loss(logits, T_val) + ml.weighted_contrastive_loss(logits.T, T_val.T))
                v_loss_cls = criterion_ce(model.lang_classifier(audio_emb), langs)

                val_loss_sum += (v_loss_cont + v_loss_cls).item()
                val_batches += 1
                
                contrast_correct += (logits.argmax(dim=1) == torch.arange(len(langs), device=ml.DEVICE)).sum().item()
                contrast_total += len(langs)
                y_true_cls.extend(langs.cpu().numpy())
                y_pred_cls.extend(model.lang_classifier(audio_emb).argmax(dim=1).cpu().numpy())

        # Metrics Calculation
        avg_train_loss = total_train_loss / train_batches
        avg_val_loss = val_loss_sum / val_batches
        overall_acc, balanced_acc, _ = ds.compute_per_class_acc(y_true_cls, y_pred_cls, args.num_lang)
        cont_acc = (100 * contrast_correct / contrast_total)

        # --- 2. UPDATE LOG.TXT ---
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{cont_acc:.2f},{overall_acc:.2f},{balanced_acc:.2f}\n")

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Class Acc: {overall_acc:.2f}%")
        
        scheduler.step(avg_val_loss)
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"dual_enc_ep{epoch}.pt"))

if __name__ == "__main__":
    train()