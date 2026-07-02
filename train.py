import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import time
import gc
import numpy as np
from sklearn.metrics import roc_auc_score

from utils.dataset import VolumeDataset
from models.end2end_pipeline import EndToEndModel

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, max_batches=None):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    start_time = time.time()
    
    for batch_idx, (volumes, labels) in enumerate(dataloader):
        volumes, labels = volumes.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training: Tự động ép kiểu Float16 để tiết kiệm 50% VRAM
        with autocast():
            # Chạy qua 3 tầng (DenoMamba -> CoreDiff -> Sybil)
            outputs = model(volumes)
            logits = outputs['logit'] # Lấy logit thô (chưa qua sigmoid) để tính Loss cho chuẩn
            
            loss = criterion(logits, labels)
            
        # Backpropagation với Scaler (do dùng Float16 nên đạo hàm rất bé, cần scale lên)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * volumes.size(0)
        
        # Tính Accuracy (Sigmoid > 0.5 coi như 1 (Cancer))
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        print(f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
        
        
        if max_batches and (batch_idx + 1) >= max_batches:
            print(f"Reached max_batches ({max_batches}). Ending epoch early to save checkpoint.")
            break
            
        # Ép dọn rác thủ công sau mỗi 10 batch để tránh phình RAM trên Kaggle
        if (batch_idx + 1) % 10 == 0:
            gc.collect()

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_preds / total_samples if total_samples > 0 else 0
    print(f"Epoch {epoch} finished in {time.time() - start_time:.2f}s | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
    return epoch_loss

def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (volumes, labels) in enumerate(dataloader):
            volumes, labels = volumes.to(device), labels.to(device)
            
            outputs = model(volumes)
            logits = outputs['logit']
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * volumes.size(0)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            del volumes, labels, outputs, logits, loss, probs, preds
            if (batch_idx + 1) % 10 == 0:
                gc.collect()
            
    val_loss = running_loss / total_samples if total_samples > 0 else 0
    val_acc = correct_preds / total_samples if total_samples > 0 else 0
    
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    
    val_auc = 0.0
    if len(np.unique(all_labels)) > 1:
        val_auc = roc_auc_score(all_labels, all_probs)
        
    print(f"--> [Validation Epoch {epoch}] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
    return val_loss, val_acc, val_auc

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Kaggle dataset directory')
    parser.add_argument('--csv_path', type=str, required=False, help='Path to NLST CSV')
    parser.add_argument('--is_nlst', action='store_true', help='Use NLST dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for AdamW')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--sybil_dropout', type=float, default=0.2, help='Dropout rate for Sybil classifier')
    parser.add_argument('--image_size', type=int, default=256, help='Image resolution (e.g. 128 or 256)')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to .pth checkpoint to resume training')
    parser.add_argument('--max_batches', type=int, default=None, help='Max batches per epoch to force early checkpoint saving')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()

    # 1. Cấu hình phần cứng
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Initializing Train/Val Dataset with Image Size {args.image_size}x{args.image_size}...")
    train_dataset = VolumeDataset(data_dir=args.data_dir, csv_file=args.csv_path, is_nlst=args.is_nlst, target_size=(args.image_size, args.image_size), split='train', val_ratio=args.val_ratio)
    val_dataset = VolumeDataset(data_dir=args.data_dir, csv_file=args.csv_path, is_nlst=args.is_nlst, target_size=(args.image_size, args.image_size), split='val', val_ratio=args.val_ratio)
    
    # Tối ưu hóa DataLoader: Set num_workers=0 để triệt tiêu hoàn toàn lỗi rò rỉ RAM (OOM) của PyTorch
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )
    
    # 3. Khởi tạo Trái tim Dự án (End-to-End)
    print("Initializing End-to-End Model...")
    model = EndToEndModel(corediff_context=True, sybil_dropout=args.sybil_dropout)
    
    # --- KỸ THUẬT KAGGLE MULTI-GPU ---
    # Nếu máy có nhiều hơn 1 GPU (Kaggle có 2xT4), tự động chia đôi Batch Size chạy song song
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs! Activating DataParallel.")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # 4. Định nghĩa Loss và Optimizer
    # Do Sybil trả về logit, ta dùng BCEWithLogitsLoss cho bài toán Binary Classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Scaler cho Mixed Precision
    scaler = GradScaler()
    
    # Tính năng Resume Training
    start_epoch = 1
    best_val_auc = 0.0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint.get('val_auc', 0.0)
        print(f"Loaded successfully. Will start from epoch {start_epoch}")
    
    # 5. Vòng lặp Huấn luyện
    print("STARTING TRAINING...")
    for epoch in range(start_epoch, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, max_batches=args.max_batches)
        
        # Lưu Checkpoint NGAY SAU KHI TRAIN xong (Phòng hờ lúc Validation bị sập RAM thì vẫn giữ được tiến độ)
        checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': 0.0 # Tạm thời để 0.0, best_model sẽ lưu AUC thực
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path} before validation")
        
        # Chấm điểm Validation
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device, epoch)
        
        # Lưu lại bản best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_path = "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': float(val_auc)
            }, best_path)
        
        # Dọn rác tổng sau mỗi Epoch
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
