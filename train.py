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
        
        # Trong lúc test local (RTX 3050), chỉ chạy 1 batch rồi break để xem code có lỗi không
        # Khi lên Kaggle sẽ comment dòng break này lại
        print(f"  -> Successfully ran 1 Batch locally, breaking to avoid OOM.")
        break 

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_preds / total_samples if total_samples > 0 else 0
    print(f"Epoch {epoch} finished in {time.time() - start_time:.2f}s | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
    return epoch_loss

def main():
    # 1. Cấu hình phần cứng
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Chuẩn bị Dữ liệu (Dùng tạm NLST hoặc AAPM cho test vòng lặp)
    # LƯU Ý: Anh cần đổi đường dẫn CSV này khớp với máy anh nhé
    data_dir = r"d:\cothuy\3Truc\National Lung Screening Trial (NLST)"
    csv_path = r"d:\cothuy\3Truc\National Lung Screening Trial (NLST)\nodule_location_globAttCRNN.csv"
    
    # Do NLST anh tải về chưa cấu trúc xong, em test tạm bằng AAPM (đã có sẵn) để lấy 1 batch
    test_dir = r"d:\cothuy\3Truc\AAPM-Mayo Clinic\LDCT-and-Projection-data"
    print("Initializing Dataset...")
    dataset = VolumeDataset(data_dir=test_dir, is_nlst=False, target_size=(256, 256))
    
    # Batch_size = 1 do GPU 3050 chỉ có 4GB-8GB VRAM
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # 3. Khởi tạo Trái tim Dự án (End-to-End)
    print("Initializing End-to-End Model...")
    model = EndToEndModel(corediff_context=True, sybil_dropout=0.2)
    
    # --- KỸ THUẬT KAGGLE MULTI-GPU ---
    # Nếu máy có nhiều hơn 1 GPU (Kaggle có 2xT4), tự động chia đôi Batch Size chạy song song
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs! Activating DataParallel.")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # 4. Định nghĩa Loss và Optimizer
    # Do Sybil trả về logit, ta dùng BCEWithLogitsLoss cho bài toán Binary Classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Scaler cho Mixed Precision
    scaler = GradScaler()
    
    # 5. Vòng lặp Huấn luyện (Chạy thử 1 Epoch)
    print("STARTING TRAINING...")
    for epoch in range(1, 2):
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
