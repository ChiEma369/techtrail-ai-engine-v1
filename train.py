import os
import sys
import argparse
import time
import json
from collections import Counter

# Đảm bảo in ký tự UTF-8 mượt mà trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models import TechTrailRNN, TechTrailLSTM

def split_into_time_steps(text, num_steps=3):
    words = str(text).split()
    if len(words) == 0:
        return [""] * num_steps
    chunk_size = max(1, len(words) // num_steps)
    chunks = []
    for i in range(num_steps):
        if i == num_steps - 1:
            chunk = words[i * chunk_size:]
        else:
            chunk = words[i * chunk_size : (i + 1) * chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

class TechTrailDataset(Dataset):
    def __init__(self, dataframe, vocab, max_seq_length=50, num_steps=3):
        self.data = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_seq_length
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text_sequence'])
        label = int(row['label'])
        time_steps = split_into_time_steps(text, self.num_steps)
        sequence_tensor = torch.zeros((self.num_steps, self.max_len), dtype=torch.long)

        for step_idx, step_text in enumerate(time_steps):
            tokens = step_text.split()
            token_ids = [self.vocab.get(word, self.vocab.get('<UNK>', 1)) for word in tokens]
            if len(token_ids) > self.max_len:
                token_ids = token_ids[:self.max_len]
            else:
                token_ids = token_ids + [self.vocab.get('<PAD>', 0)] * (self.max_len - len(token_ids))
            sequence_tensor[step_idx] = torch.tensor(token_ids, dtype=torch.long)

        return sequence_tensor, torch.tensor(label, dtype=torch.long)

def build_vocab(texts, min_freq=1):
    word_counts = Counter()
    for text in texts:
        word_counts.update(str(text).split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.items():
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_x.size(0)

    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return total_loss / total_samples, total_correct / total_samples, np.array(all_preds), np.array(all_labels)

def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='royalblue', marker='o')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', color='crimson', marker='s')
    axes[0].set_title('Loss Convergence Curve', fontsize=13)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=11)

    # Accuracy Curve
    axes[1].plot(epochs, [a * 100 for a in history['train_acc']], label='Train Acc (%)', color='royalblue', marker='o')
    axes[1].plot(epochs, [a * 100 for a in history['val_acc']], label='Val Acc (%)', color='forestgreen', marker='s')
    axes[1].set_title('Accuracy Curve across Epochs', fontsize=13)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved training curves to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình RNN/LSTM cho TechTrail AI Engine")
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'rnn'], help="Kiểu mô hình (lstm / rnn)")
    parser.add_argument('--epochs', type=int, default=15, help="Số epoch huấn luyện tối đa")
    parser.add_argument('--lr', type=float, default=2e-3, help="Tốc độ học (learning rate)")
    parser.add_argument('--batch_size', type=int, default=32, help="Kích thước mini-batch")
    parser.add_argument('--embed_dim', type=int, default=128, help="Chiều vector nhúng từ (embedding dimension)")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Chiều ẩn của RNN/LSTM")
    parser.add_argument('--num_layers', type=int, default=1, help="Số lớp RNN/LSTM xếp chồng")
    parser.add_argument('--dropout', type=float, default=0.3, help="Tỷ lệ dropout")
    parser.add_argument('--patience', type=int, default=5, help="Số epoch chờ trước khi early stopping")
    parser.add_argument('--data_dir', type=str, default='data', help="Thư mục chứa dữ liệu CSV")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help="Thư mục lưu mô hình và kết quả")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING INITIALIZATION: MODEL {args.model.upper()} ON DEVICE [{device}] ===")

    # 1. Đọc dữ liệu
    train_path = os.path.join(args.data_dir, 'train.csv')
    val_path = os.path.join(args.data_dir, 'val.csv')
    test_path = os.path.join(args.data_dir, 'test.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

    # Ánh xạ nhãn
    label_names = sorted(train_df['category'].unique())
    num_classes = len(label_names)
    print(f"Number of target classes: {num_classes}")

    # 2. Xây dựng Vocab & DataLoaders
    vocab = build_vocab(train_df['text_sequence'])
    print(f"Vocabulary size: {len(vocab)} words")

    # Lưu vocab
    vocab_path = os.path.join(args.save_dir, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    train_dataset = TechTrailDataset(train_df, vocab, max_seq_length=50, num_steps=3)
    val_dataset = TechTrailDataset(val_df, vocab, max_seq_length=50, num_steps=3)
    test_dataset = TechTrailDataset(test_df, vocab, max_seq_length=50, num_steps=3)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Khởi tạo mô hình
    if args.model == 'lstm':
        model = TechTrailLSTM(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    else:
        model = TechTrailRNN(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 4. Huấn luyện
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.save_dir, f'best_{args.model}_model.pt')
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    start_time = time.time()
    print("\n" + "="*70)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<11} | {'Val Acc':<10} | {'Status'}")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        scheduler.step(v_loss)

        status = ""
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), best_model_path)
            status = "(*) Saved Best"
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"{epoch:02d}/{args.epochs:02d}    | {t_loss:<12.4f} | {t_acc*100:<10.2f}% | {v_loss:<11.4f} | {v_acc*100:<9.2f}% | {status}")

        # Kiểm tra hội tụ qua Early Stopping
        if patience_counter >= args.patience:
            print(f"\n[INFO] Convergence reached (val_loss did not improve for {args.patience} epochs). Early stopping at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    print("="*70)
    print(f"Total training time: {total_time:.2f} seconds")

    # 5. Lưu biểu đồ huấn luyện
    plot_path = os.path.join(args.save_dir, f'{args.model}_training_curves.png')
    plot_curves(history, plot_path)

    # 6. Đánh giá trên tập Test
    print(f"\n=== TEST EVALUATION ({len(test_loader.dataset)} unseen test samples) ===")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"-> Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=label_names, digits=4))

    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    main()
