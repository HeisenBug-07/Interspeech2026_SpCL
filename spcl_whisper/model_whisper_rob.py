import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaModel, RobertaTokenizer, WhisperModel, WhisperFeatureExtractor

# --- Constants & Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WHISPER_MODEL = "openai/whisper-base"
ROBERTA_MODEL = "FacebookAI/roberta-base"

# --- RoBERTa Text Encoder Wrapper ---
class RobertaTextEncoderWrapper(nn.Module):
    def __init__(self, model_name=ROBERTA_MODEL, projection_dim=512, device=DEVICE):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name).to(device)
        
        # 1. Freeze all RoBERTa parameters first
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze the last 4 layers
        num_layers = len(self.roberta.encoder.layer) # 12
        layers_to_unfreeze = 4
        
        for i in range(num_layers - layers_to_unfreeze, num_layers):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = True
                
        if self.roberta.pooler is not None:
            for param in self.roberta.pooler.parameters():
                param.requires_grad = True

        self.hidden_size = self.roberta.config.hidden_size
        self.projection = nn.Linear(self.hidden_size, projection_dim).to(device)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = out.last_hidden_state[:, 0, :]
        return self.projection(cls_embeddings)

# --- Whisper Audio Encoder Wrapper (Updated for Base) ---
class WhisperAudioEncoderWrapper(nn.Module):
    def __init__(self, model_name=WHISPER_MODEL, projection_dim=512, device=DEVICE):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name).to(device)
        self.encoder = self.whisper.encoder
        
        # 1. Freeze the entire Decoder
        for param in self.whisper.decoder.parameters():
            param.requires_grad = False
            
        # 2. Freeze all Encoder layers first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 3. Unfreeze the last 2 layers (Base has 6 layers total)
        num_layers = len(self.encoder.layers) 
        layers_to_unfreeze = 2
        
        for i in range(num_layers - layers_to_unfreeze, num_layers):
            for param in self.encoder.layers[i].parameters():
                param.requires_grad = True
        
        # 4. Final layer norm
        for param in self.encoder.layer_norm.parameters():
            param.requires_grad = True

        self.hidden_size = self.whisper.config.d_model
        self.projection = nn.Linear(self.hidden_size, projection_dim).to(device)

    def forward(self, input_features):
        outputs = self.encoder(input_features)
        hidden_states = outputs.last_hidden_state 
        pooled_output = hidden_states.mean(dim=1)
        return self.projection(pooled_output)

# --- Main Dual Encoder Model ---
class AudioTextDualEncoder(nn.Module):
    def __init__(self, num_lang, device=DEVICE):
        super().__init__()
        self.device = device
        self.audio_branch = WhisperAudioEncoderWrapper(device=device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
        self.text_branch = RobertaTextEncoderWrapper(device=device)
        self.tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.lang_classifier = nn.Linear(512, num_lang).to(device)

    def get_logit_scale(self):
        return self.logit_scale.exp()

# --- Builder and Utilities ---
def build_model(num_lang, device=DEVICE):
    print(f"Building Dual Encoder (RoBERTa + Whisper Base) on {device}...")
    model = AudioTextDualEncoder(num_lang, device)
    return model.to(device)

def encode_audio(model, batch_wavforms):
    wavs_np = [w.cpu().numpy() if isinstance(w, torch.Tensor) else w for w in batch_wavforms]
    inputs = model.feature_extractor(
        wavs_np, 
        sampling_rate=16000, 
        return_tensors="pt",
        padding="max_length", 
        truncation=True
    ).to(DEVICE)
    return model.audio_branch(inputs.input_features)

def encode_text_prompts(model, prompts):
    tokenized = model.tokenizer(
        prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(DEVICE)
    return model.text_branch(tokenized["input_ids"], tokenized["attention_mask"])

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
