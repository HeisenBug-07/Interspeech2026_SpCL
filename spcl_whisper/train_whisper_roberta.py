import os, argparse, random, torch, torch.nn as nn, numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import dataloader as ds
import model_whisper_rob as ml 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", type=str, default="train_ekstep_md.txt")
    parser.add_argument("--val_txt", type=str, default="val_ekstep_md.txt")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt_whisper_base_roberta_0.5") 
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--lr", type=float, default=5e-5)     
    parser.add_argument("--num_lang", type=int, default=12)
    parser.add_argument("--contrast_weight", type=float, default=0.3)
    parser.add_argument("--class_weight", type=float, default=0.7)
    parser.add_argument("--sr", type=int, default=16000)
    return parser.parse_args()

def train():
    args = get_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # --- 1. INITIALIZE LOG FILE ---
    log_path = os.path.join(args.ckpt_dir, "log_whisper_base_roberta_0.5.txt")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,contrast_acc,class_acc,balanced_acc\n")

    model = ml.build_model(args.num_lang)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    criterion_ce = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        ds.LIDDataset(args.train_txt, sr=args.sr), 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=ds.collate_fn,
        num_workers=8,        # CPUs used for audio processing
        pin_memory=True       # Speeds up host-to-device transfer
    )

    val_loader = DataLoader(
        ds.LIDDataset(args.val_txt, sr=args.sr), 
        batch_size=args.batch_size, 
        collate_fn=ds.collate_fn,
        num_workers=8,
        pin_memory=True
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for wavs, caps, langs, doms in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            langs, doms = langs.to(ml.DEVICE), doms.to(ml.DEVICE)
            optimizer.zero_grad()
            
            a_emb = ml.encode_audio(model, wavs)
            t_emb = ml.encode_text_prompts(model, caps)
            logits = ml.compute_similarity_logits(a_emb, t_emb, model)
            T = ml.build_target_matrix(langs, doms, ml.DEVICE)

            loss_con = 0.5 * (ml.weighted_contrastive_loss(logits, T) + ml.weighted_contrastive_loss(logits.T, T.T))
            loss_cls = criterion_ce(model.lang_classifier(a_emb), langs)
            
            loss = (args.contrast_weight * loss_con) + (args.class_weight * loss_cls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss_sum, con_corr, total = 0, 0, 0
        preds, truths = [], []
        
        with torch.no_grad():
            for wavs, caps, langs, doms in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                langs, doms = langs.to(ml.DEVICE), doms.to(ml.DEVICE)
                a_emb = ml.encode_audio(model, wavs)
                t_emb = ml.encode_text_prompts(model, caps)
                logits = ml.compute_similarity_logits(a_emb, t_emb, model)
                T_val = ml.build_target_matrix(langs, doms, ml.DEVICE)
                
                v_loss_con = 0.5 * (ml.weighted_contrastive_loss(logits, T_val) + ml.weighted_contrastive_loss(logits.T, T_val.T))
                cls_out = model.lang_classifier(a_emb)
                v_loss_cls = criterion_ce(cls_out, langs)

                val_loss_sum += (v_loss_con + v_loss_cls).item()
                con_corr += (logits.argmax(dim=1) == torch.arange(len(langs)).to(ml.DEVICE)).sum().item()
                total += len(langs)
                preds.extend(cls_out.argmax(dim=1).cpu().numpy())
                truths.extend(langs.cpu().numpy())

        # --- Metrics Calculation ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        con_acc = 100 * con_corr / total
        class_acc, balanced_acc, _ = ds.compute_per_class_acc(truths, preds, args.num_lang)

        # --- 2. APPEND TO LOG FILE ---
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{con_acc:.2f},{class_acc:.2f},{balanced_acc:.2f}\n")

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Class Acc: {class_acc:.2f}%")
        
        torch.save(model.state_dict(), f"{args.ckpt_dir}/model_ep{epoch}.pt")
        scheduler.step(avg_val_loss)

if __name__ == "__main__":
    train()
