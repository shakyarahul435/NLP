import torch
import torch.nn as nn
import math
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import os
import pickle

# Model parameters
emb_dim = 256
hid_dim = 256
num_layers = 2
dropout_rate = 0.3

# Load vocab
vocab_path = os.path.join(settings.BASE_DIR, 'model', 'vocab.pkl')
itos_path = os.path.join(settings.BASE_DIR, 'model', 'itos.pkl')
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
with open(itos_path, 'rb') as f:
    itos = pickle.load(f)
vocab_size = len(vocab)

# Tokenizer
def nepali_tokenizer(text):
    return text.strip().split()

# Model class
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim) 
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            getattr(self.lstm, f'weight_ih_l{i}').data.uniform_(-init_range_other, init_range_other)
            getattr(self.lstm, f'weight_hh_l{i}').data.uniform_(-init_range_other, init_range_other)
            getattr(self.lstm, f'bias_ih_l{i}').data.zero_()
            getattr(self.lstm, f'bias_hh_l{i}').data.zero_()
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
        
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden):
        embedding = self.embedding(src)
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

# Load model
device = torch.device('cpu')
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model_path = os.path.join(settings.BASE_DIR, 'model', 'best-model.pt')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Generate function
def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed):
    torch.manual_seed(seed)
    model.eval()
    
    tokens = tokenizer(prompt)
    input_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    
    hidden = model.init_hidden(1, device)
    
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_seq_len):
            prediction, hidden = model(input_tensor, hidden)
            prediction = prediction[:, -1, :]
            
            prediction = prediction / temperature
            
            probs = torch.softmax(prediction, dim=-1)
            
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            next_token = itos[next_token_id]
            
            generated_tokens.append(next_token)
            
            input_tensor = torch.LongTensor([[next_token_id]]).to(device)
    
    return ' '.join(generated_tokens)

# Transliteration function
def transliterate(text):
    # Simple mapping for demo
    mapping = {
        'hamra': 'हाम्रा',
        'nepal': 'नेपाल',
        'ho': 'हो',
        'ek': 'एक',
        'sundar': 'सुन्दर',
        'desh': 'देश',
        # Add more as needed
    }
    words = text.split()
    transliterated_words = [mapping.get(word.lower(), word) for word in words]
    return ' '.join(transliterated_words)

@api_view(['POST'])
def generate_text(request):
    prompt = request.data.get('prompt', '')
    max_len = int(request.data.get('max_len', 20))
    temperature = float(request.data.get('temperature', 0.5))
    seed = int(request.data.get('seed', 42))
    
    print(f"Received prompt: {prompt}, max_len: {max_len}, temperature: {temperature}")
    generated = generate(prompt, max_len, temperature, model, nepali_tokenizer, vocab, device, seed)
    print(f"Generated text: {generated}")
    return Response({'generated': generated})

@api_view(['POST'])
def transliterate_text(request):
    text = request.data.get('text', '')
    transliterated = transliterate(text)
    return Response({'transliterated': transliterated})
