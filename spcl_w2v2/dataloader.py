import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from collections import Counter

class LIDDataset(Dataset):
    def __init__(self, txt_file, sr=16000):
        self.items =[]
        self.sr = sr
        if not os.path.exists(txt_file):
            print(f"{txt_file} does not exist.")
            return
        
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # Parsing logic for path "caption" label domain
                    first_quote = line.index('"')
                    last_quote = line.rindex('"')
                    prefix = line[:first_quote].strip()
                    caption = line[first_quote+1:last_quote]
                    suffix = line[last_quote+1:].strip()
                    path = prefix.split()[0]
                    parts = suffix.split()
                    class_label = int(parts[0])
                    domain_label = int(parts[1]) if len(parts) > 1 else 0
                except ValueError:
                    # Fallback parsing
                    parts = line.split()
                    path = parts[0]
                    caption = " ".join(parts[1:-2]).strip('"') if len(parts) > 2 else ""
                    class_label = int(parts[-2])
                    domain_label = int(parts[-1])
                self.items.append((path, caption, class_label, domain_label))

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path, caption, label, domain = self.items[idx]
        try:
            wav, orig_sr = torchaudio.load(path)
        except Exception as e:
            print(f"Error loading {path} with torchaudio: {e}")
            # Attempt fallbacks: soundfile (pysoundfile) then librosa
            wav = None
            orig_sr = self.sr
            try:
                import soundfile as sf
                data, orig_sr = sf.read(path)
                data = np.asarray(data)
                if data.ndim == 1:
                    wav = torch.from_numpy(data).unsqueeze(0).float()
                else:
                    # soundfile returns (nsamples, channels)
                    wav = torch.from_numpy(data.T).float()
                    # Keep behavior consistent: convert to mono
                    if wav.ndim > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                print(f"Loaded {path} with soundfile (sr={orig_sr})")
            except Exception as e2:
                print(f" - soundfile failed: {e2}")
                try:
                    import librosa
                    data, orig_sr = librosa.load(path, sr=None, mono=False)
                    data = np.asarray(data)
                    # librosa returns (nsamples,) for mono or (n_channels, nsamples) if mono=False
                    if data.ndim == 1:
                        wav = torch.from_numpy(data).unsqueeze(0).float()
                    else:
                        # Ensure shape (channels, samples)
                        if data.shape[0] > data.shape[1]:
                            # (channels, samples) already
                            wav = torch.from_numpy(data).float()
                        else:
                            # (samples, channels) -> transpose
                            wav = torch.from_numpy(data.T).float()
                        if wav.ndim > 1:
                            wav = wav.mean(dim=0, keepdim=True)
                    print(f"Loaded {path} with librosa (sr={orig_sr})")
                except Exception as e3:
                    print(f" - librosa failed: {e3}")
                    print(f"Falling back to silence for {path}")
                    wav = torch.zeros(1, self.sr)
                    orig_sr = self.sr

        if wav is None:
            wav = torch.zeros(1, self.sr)

        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if orig_sr != self.sr:
            wav = torchaudio.functional.resample(wav, orig_sr, self.sr)
        return wav.squeeze(0), caption, int(label), int(domain)
    
def collate_fn(batch):
    wavs, caps, langs, doms = zip(*batch)
    lens = [w.shape[0] for w in wavs]
    max_len = max(lens)
    padded = []
    for w in wavs:
        if w.shape[0] < max_len:
            w = w.repeat((max_len //len(w) +1))[:max_len]

        padded.append(w)

    wav_batch = torch.stack(padded)
    return wav_batch, list(caps), torch.tensor(langs, dtype=torch.long), torch.tensor(doms, dtype=torch.long)

def compute_class_weights(txt_file, num_classes):
    labels = []
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                parts = line.split()
                label = int(parts[-2])
                labels.append(label)
            except:
                continue

    if not labels: return None

    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes)
    
    for c in range(num_classes):
        count = counts.get(c, 0)
        weights[c] = total / (num_classes * (count + 1e-6))
    
    return weights

def compute_per_class_acc(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    per_class = {}
    for c in range(num_classes):
        idx = (y_true == c)
        per_class[c] = (y_pred[idx] == c).mean() * 100.0 if idx.sum() else 0.0
    overall = (y_true == y_pred).mean() * 100.0
    balanced = np.mean(list(per_class.values()))
    return overall, balanced, per_class