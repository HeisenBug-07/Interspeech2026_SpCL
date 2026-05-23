import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaModel, RobertaTokenizer, Wav2Vec2Model

# --- Constants & Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- RoBERTa Text Encoder Wrapper ---
class RobertaTextEncoderWrapper(nn.Module):
    def __init__(self, model_name="FacebookAI/roberta-base", projection_dim=512, device=DEVICE):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name).to(device)
        self.hidden_size = self.roberta.config.hidden_size
        self.projection = nn.Linear(self.hidden_size, projection_dim).to(device)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Using [CLS] token (index 0) for RoBERTa
        cls_embeddings = out.last_hidden_state[:, 0, :]
        return self.projection(cls_embeddings)

# --- Wav2Vec2 Audio Encoder Wrapper ---
class Wav2Vec2AudioEncoderWrapper(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-lv-60-espeak-cv-ft", projection_dim=512, device=DEVICE):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.hidden_size = self.wav2vec2.config.hidden_size
        self.projection = nn.Linear(self.hidden_size, projection_dim).to(device)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        return self.projection(pooled_output)

# --- Main Dual Encoder Model ---
class AudioTextDualEncoder(nn.Module):
    def __init__(self, num_lang, device=DEVICE):
        super().__init__()
        self.device = device
        self.audio_branch = Wav2Vec2AudioEncoderWrapper(device=device)
        self.text_branch = RobertaTextEncoderWrapper(device=device)
        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.lang_classifier = nn.Linear(512, num_lang).to(device)

    def get_logit_scale(self):
        return self.logit_scale.exp()

# --- THE MISSING BUILDER FUNCTION ---
def build_model(num_lang, device=DEVICE):
    print(f"Building Dual Encoder (RoBERTa + Wav2Vec2) on {device}...")
    model = AudioTextDualEncoder(num_lang, device)
    model.to(device)
    return model

# --- Utility Functions for Training Script ---

def encode_audio(model, batch_wavforms, max_len=None):
    if max_len is None:
        max_len = max([w.shape[0] for w in batch_wavforms])

    padded_wavs = []
    for w in batch_wavforms:
        if w.shape[0] > max_len:
            padded_wavs.append(w[:max_len])
        else:
            padding = torch.zeros(max_len - w.shape[0], device=w.device)
            padded_wavs.append(torch.cat((w, padding)))
            
    input_tensor = torch.stack(padded_wavs).to(DEVICE)
    return model.audio_branch(input_tensor)

def encode_text_prompts(model, prompts):
    tokenized = model.tokenizer(
        prompts, 
        padding=True, 
        truncation=True, 
        max_length=154, 
        return_tensors="pt"
    ).to(DEVICE)
    
    return model.text_branch(
        input_ids=tokenized["input_ids"], 
        attention_mask=tokenized["attention_mask"]
    )

def build_target_matrix(lang_ids, dom_ids, device):
    B = lang_ids.shape[0]
    lang_i, lang_j = lang_ids.unsqueeze(1), lang_ids.unsqueeze(0)
    dom_i, dom_j = dom_ids.unsqueeze(1), dom_ids.unsqueeze(0)
    same_lang = (lang_i == lang_j)
    same_dom = (dom_i == dom_j)
    T = torch.zeros((B, B), device=device)
    T[same_lang & same_dom] = 1.0
    T[same_lang & (~same_dom)] = 0.5
    T.fill_diagonal_(1.0)
    return T

def compute_similarity_logits(audio_emb, text_emb, model):
    audio_norm = audio_emb / audio_emb.norm(dim=1, keepdim=True)
    text_norm = text_emb / text_emb.norm(dim=1, keepdim=True)
    return model.get_logit_scale() * (audio_norm @ text_norm.t())

def weighted_contrastive_loss(logits, targets_T):
    log_p = torch.log_softmax(logits, dim=1)
    return (- (targets_T * log_p).sum(dim=1)).mean()
