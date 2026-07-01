import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import time

from utils.dataset import VolumeDataset
from models.end2end_pipeline import EndToEndModel

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch):
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
        


    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_preds / total_samples if total_samples > 0 else 0
    print(f"Epoch {epoch} finished in {time.time() - start_time:.2f}s | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
    return epoch_loss

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
    args = parser.parse_args()

    # 1. Cấu hình phần cứng
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Initializing Dataset with Image Size {args.image_size}x{args.image_size}...")
    dataset = VolumeDataset(data_dir=args.data_dir, csv_file=args.csv_path, is_nlst=args.is_nlst, target_size=(args.image_size, args.image_size))
    
    # Tối ưu hóa DataLoader: tăng num_workers, thêm pin_memory để copy thẳng dữ liệu vào RAM GPU, thêm prefetch_factor
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2
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
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded successfully. Will start from epoch {start_epoch}")
    
    # 5. Vòng lặp Huấn luyện
    print("STARTING TRAINING...")
    for epoch in range(start_epoch, args.epochs + 1):
        train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch)
        
        # Lưu Checkpoint chống Timeout trên Kaggle
        checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main()
