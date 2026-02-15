"""API views for the sentence similarity backend."""
from __future__ import annotations

import json
from typing import Any, Dict

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

try:  # pragma: no cover - heavy dependencies
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertModel, BertTokenizer, pipeline
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The transformers and torch packages are required. Install them with 'pip install torch transformers'."
    ) from exc


DEVICE = torch.device("cpu")  # Force CPU due to CUDA compatibility issues
MODEL_NAME = "bert-base-uncased"
_TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME)
_MODEL = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)

import os

# ... existing code ...

# Load saved model from A4_2.ipynb (fine-tuned BERT for sentence similarity)
saved_model_loaded = False
if os.path.exists('../../model/bert_sentence.pth'):
    try:
        state_dict = torch.load('../../model/bert_sentence.pth', map_location='cpu')[0]
        _MODEL.load_state_dict(state_dict)
        saved_model_loaded = True
    except Exception:
        pass

_MODEL.eval()

# Load fill-mask pipeline
_MASK_PIPELINE = pipeline('fill-mask', model=MODEL_NAME, device=DEVICE)


# Custom BERT classes from A4.ipynb
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model)      # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        #x, seg: (bs, len)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(self.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (len,) -> (bs, len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn 


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.device = device
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(self.n_heads * self.d_v, self.d_model, device=self.device)(context)
        return nn.LayerNorm(self.d_model, device=self.device)(output + residual), attn # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)
        self.pos_ffn       = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class BERT(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device):
        super(BERT, self).__init__()
        self.params = {'n_layers': n_layers, 'n_heads': n_heads, 'd_model': d_model,
                       'd_ff': d_ff, 'd_k': d_k, 'n_segments': n_segments,
                       'vocab_size': vocab_size, 'max_len': max_len}
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, d_k, device) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device = device

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        
        # 1. predict next sentence
        # it will be decided by first token(CLS)
        h_pooled   = self.activ(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_nsp = self.classifier(h_pooled) # [batch_size, 2]

        # 2. predict the masked token
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked  = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_nsp
    
    def get_last_hidden_state(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        return output


# Load custom BERT from A4.ipynb
_CUSTOM_MODEL = None
_WORD2IDX = None
_IDX2WORD = None
if os.path.exists('../../model/model_bert.pt'):
    try:
        params, state, word2idx, idx2word = torch.load('../../model/model_bert.pt')
        _CUSTOM_MODEL = BERT(**params, device=DEVICE).to(DEVICE)
        _CUSTOM_MODEL.load_state_dict(state)
        _WORD2IDX = word2idx
        _IDX2WORD = idx2word
        _CUSTOM_MODEL.eval()
    except Exception as e:
        print(f"Failed to load custom model: {e}")
        _CUSTOM_MODEL = None


def _mean_pool(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    pooled = (embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled


def _encode(sentence_a: str, sentence_b: str) -> torch.Tensor:
    encoded = _TOKENIZER([sentence_a, sentence_b], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model_output = _MODEL(**encoded)
    pooled = _mean_pool(model_output.last_hidden_state, encoded["attention_mask"])
    normalized = F.normalize(pooled, p=2, dim=1)
    return normalized


@csrf_exempt
@require_http_methods(["POST"])
def masked_prediction(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body or "{}")
        sentence = payload["sentence"].strip()
    except (KeyError, AttributeError, json.JSONDecodeError) as exc:
        return JsonResponse({"error": "Provide sentence with [MASK]."}, status=400)

    if "[MASK]" not in sentence:
        return JsonResponse({"error": "Sentence must contain [MASK]."}, status=400)

    # Use the fill-mask pipeline
    predictions = _MASK_PIPELINE(sentence)

    # Format the response
    result = {
        "sentence": sentence,
        "predictions": [
            {
                "token_str": pred["token_str"],
                "score": pred["score"]
            } for pred in predictions
        ]
    }

    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["GET"])
def health_check(request: HttpRequest) -> JsonResponse:
    data: Dict[str, Any] = {
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "loaded_saved_model": saved_model_loaded,
        "custom_model_loaded": _CUSTOM_MODEL is not None,
    }
    return JsonResponse(data)


@csrf_exempt
@require_http_methods(["POST"])
def sentence_similarity(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body or "{}")
        sentence_a = payload["sentence_a"].strip()
        sentence_b = payload["sentence_b"].strip()
    except (KeyError, AttributeError, json.JSONDecodeError) as exc:
        return JsonResponse({"error": "Provide sentence_a and sentence_b."}, status=400)

    embeddings = _encode(sentence_a, sentence_b)
    similarity = torch.dot(embeddings[0], embeddings[1]).item()

    return JsonResponse(
        {
            "sentence_a": sentence_a,
            "sentence_b": sentence_b,
            "similarity": similarity,
        }
    )
