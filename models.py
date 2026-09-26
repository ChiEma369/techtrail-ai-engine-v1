import torch
import torch.nn as nn

class TechTrailRNN(nn.Module):
    """
    Mô hình mạng nơ-ron đệ quy cơ bản (Simple RNN) cho bài toán phân loại định hướng nghề nghiệp.
    Xử lý chuỗi gồm `num_steps` bước phản hồi (mỗi bước là một đoạn văn bản có tối đa `max_len` tokens).
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=8, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Simple RNN Layer
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Classifier Head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Input x: (batch_size, num_steps, max_len)
        Output logits: (batch_size, num_classes)
        """
        batch_size, num_steps, max_len = x.shape
        
        # 1. Embedding từng token: (B, S, L, D)
        embeds = self.embedding(x)
        
        # 2. Mean pooling trên các token hợp lệ (loại trừ padding_idx=0) để lấy vector đại diện từng bước
        mask = (x != 0).unsqueeze(-1).float() # (B, S, L, 1)
        sum_embeds = (embeds * mask).sum(dim=2) # (B, S, D)
        lengths = mask.sum(dim=2).clamp(min=1e-9) # (B, S, 1)
        step_features = sum_embeds / lengths # (B, S, D)
        step_features = self.dropout(step_features)
        
        # 3. Chuyển qua mạng Simple RNN theo chuỗi thời gian
        # out: (B, S, hidden_dim), hn: (num_layers, B, hidden_dim)
        out, hn = self.rnn(step_features)
        
        # Lấy hidden state của bước cuối cùng
        last_hidden = hn[-1] # (B, hidden_dim)
        
        # 4. Phân loại nghề nghiệp
        logits = self.fc(last_hidden)
        return logits


class TechTrailLSTM(nn.Module):
    """
    Mô hình LSTM (Long Short-Term Memory) cho bài toán phân loại định hướng nghề nghiệp.
    Giải quyết suy giảm gradient và ghi nhớ ngữ cảnh dài hạn qua các bước phản hồi.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=8, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Classifier Head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Input x: (batch_size, num_steps, max_len)
        Output logits: (batch_size, num_classes)
        """
        batch_size, num_steps, max_len = x.shape
        
        # 1. Embedding từng token: (B, S, L, D)
        embeds = self.embedding(x)
        
        # 2. Mean pooling trên các token hợp lệ (loại trừ padding_idx=0)
        mask = (x != 0).unsqueeze(-1).float() # (B, S, L, 1)
        sum_embeds = (embeds * mask).sum(dim=2) # (B, S, D)
        lengths = mask.sum(dim=2).clamp(min=1e-9) # (B, S, 1)
        step_features = sum_embeds / lengths # (B, S, D)
        step_features = self.dropout(step_features)
        
        # 3. Truyền qua LSTM theo chuỗi các bước phản hồi
        # out: (B, S, hidden_dim), hn: (num_layers, B, hidden_dim), cn: (num_layers, B, hidden_dim)
        out, (hn, cn) = self.lstm(step_features)
        
        # Lấy hidden state của bước cuối cùng
        last_hidden = hn[-1] # (B, hidden_dim)
        
        # 4. Phân loại nghề nghiệp
        logits = self.fc(last_hidden)
        return logits
