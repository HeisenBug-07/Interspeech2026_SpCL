import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

# Import your existing modules
import dataloader as ds
import model_whisper_rob as ml

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluation Script for Whisper-Base-RoBERTa Dual Encoder")
    parser.add_argument("--eval_txt", type=str, required=True, help="Path to eval.txt")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the .pt file")
    parser.add_argument("--num_lang", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32) # Increased for Whisper-Base
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    # 1. Load Model
    print(f"Initializing Whisper-Base Dual Encoder ({args.num_lang} languages)...")
    model = ml.build_model(args.num_lang)
    
    print(f"Loading weights from {args.ckpt_path}...")
    state_dict = torch.load(args.ckpt_path, map_location=ml.DEVICE)
    model.load_state_dict(state_dict)
    model.to(ml.DEVICE)
    model.eval()

    # 2. Prepare Data
    eval_ds = ds.LIDDataset(args.eval_txt, sr=args.sr)
    eval_loader = DataLoader(
        eval_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=ds.collate_fn, 
        num_workers=4
    )

    y_true = []
    y_pred = []
    con_correct = 0
    total_samples = 0

    # 3. Inference Loop
    print(f"Running inference on {ml.DEVICE}...")
    with torch.no_grad():
        for wavs, caps, langs, _ in tqdm(eval_loader, desc="Evaluating"):
            langs = langs.to(ml.DEVICE)
            
            # Encode both branches
            audio_emb = ml.encode_audio(model, wavs)
            text_emb = ml.encode_text_prompts(model, caps)
            
            # 1. Contrastive Performance (Audio-Text Alignment)
            logits = ml.compute_similarity_logits(audio_emb, text_emb, model)
            con_preds = torch.argmax(logits, dim=1)
            # Contrastive accuracy assumes diagonal is the correct pair in a batch
            con_correct += (con_preds == torch.arange(len(langs), device=ml.DEVICE)).sum().item()
            
            # 2. Classification Performance (Language ID)
            cls_logits = model.lang_classifier(audio_emb)
            cls_preds = torch.argmax(cls_logits, dim=1)
            
            y_true.extend(langs.cpu().numpy())
            y_pred.extend(cls_preds.cpu().numpy())
            total_samples += len(langs)

    # 4. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    contrastive_acc = (con_correct / total_samples) * 100
    report = classification_report(y_true, y_pred, digits=4)

    # 5. Final Printout
    print("\n" + "="*60)
    print("FINAL EVALUATION METRICS (WHISPER-BASE)")
    print("="*60)
    print(f"Overall Class Accuracy (LID):     {acc * 100:.2f}%")
    print(f"Balanced Class Accuracy:          {balanced_acc * 100:.2f}%")
    print(f"Contrastive Match Accuracy:       {contrastive_acc:.2f}%")
    print("-" * 60)
    print("Per-Class Classification Stats:")
    print(report)
    print("="*60)

if __name__ == "__main__":
    evaluate()
