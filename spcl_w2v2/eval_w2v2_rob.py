import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

# Custom imports from your provided training structure
import dataloader as ds
import model_w2v2_roberta as ml

def get_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for Wav2Vec2-RoBERTa Dual Encoder")
    
    # Required Inputs
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained .pt checkpoint")
    parser.add_argument("--eval_txt", type=str, required=True, help="Path to the evaluation metadata file")
    
    # Model/Data Params (Should match training config)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_lang", type=int, default=12)
    parser.add_argument("--max_wav_len", type=int, default=160000)
    parser.add_argument("--sr", type=int, default=16000)
    
    return parser.parse_args()

def run_evaluation():
    args = get_eval_args()

    # 1. Initialize Model Architecture
    print(f"--- Initializing Model ({args.num_lang} languages) ---")
    model = ml.build_model(args.num_lang)

    # 2. Load Checkpoint Weights
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint not found at {args.ckpt_path}")
        return

    print(f"--- Loading weights from {args.ckpt_path} ---")
    state_dict = torch.load(args.ckpt_path, map_location=ml.DEVICE)
    model.load_state_dict(state_dict)
    model.to(ml.DEVICE)
    model.eval()

    # 3. Prepare Data Loader
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

    # 4. Inference Loop
    print("--- Running Inference ---")
    with torch.no_grad():
        for wavs, _, langs, _ in tqdm(eval_loader, desc="Processing Audio"):
            wavs = wavs.to(ml.DEVICE)
            
            # Feature extraction and classification
            audio_emb = ml.encode_audio(model, wavs, args.max_wav_len)
            logits = model.lang_classifier(audio_emb)
            preds = torch.argmax(logits, dim=1)

            # Collect results
            y_true.extend(langs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 5. Corrected Metrics Calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall Accuracy
    overall_acc = accuracy_score(y_true, y_pred) * 100

    # Balanced Accuracy (Arithmetic mean of class-specific recall)
    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100

    # Per-Class Accuracy Calculation
    # We use a confusion matrix to get True Positives (diagonal) 
    # and Total Samples per class (row sum)
    cm = confusion_matrix(y_true, y_pred, labels=range(args.num_lang))
    
    # Avoid division by zero if a class has no samples in the eval file
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    # 6. Print Formatted Results
    print("\n" + "="*40)
    print(f"{'EVALUATION REPORT':^40}")
    print("="*40)
    print(f"Overall Accuracy:  {overall_acc:>10.2f}%")
    print(f"Balanced Accuracy: {balanced_acc:>10.2f}%")
    print("-" * 40)
    print(f"{'Language ID':<15} | {'Accuracy':>15}")
    print("-" * 40)
    
    for idx, acc in enumerate(per_class_acc):
        if np.isnan(acc):
            print(f"Language {idx:02d}    |      No Samples")
        else:
            print(f"Language {idx:02d}    | {acc:>14.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()