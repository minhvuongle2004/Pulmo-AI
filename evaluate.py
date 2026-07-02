import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import numpy as np

from utils.dataset import VolumeDataset
from models.end2end_pipeline import EndToEndModel

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Đang tiến hành chấm điểm mô hình (Inference)...")
    
    with torch.no_grad():
        for batch_idx, (volumes, labels) in enumerate(dataloader):
            volumes, labels = volumes.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(volumes)
            logits = outputs['logit']
            
            # Sigmoid để ra xác suất 0 -> 1
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Lưu lại để tính toán
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            print(f"Batch {batch_idx+1}/{len(dataloader)}:")
            for i in range(len(probs)):
                lbl = int(labels[i].item())
                prb = probs[i].item()
                prd = int(preds[i].item())
                print(f"  - Bệnh nhân {i+1}: AI đoán {prb:.4f} (Nhãn AI: {prd}) | Đáp án thật: {lbl}")
                
            # Đọc khoảng 20 batch (~80 bệnh nhân) là đủ để đánh giá nhanh, nếu muốn chạy hết thì bỏ comment dòng break
            if batch_idx > 20:
                print("Đã quét đủ số lượng bệnh nhân đại diện. Dừng chấm điểm để báo cáo.")
                break
                
    # Chuyển sang numpy array
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Tính toán các chỉ số Y Khoa
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n[KẾT QUẢ TỔNG QUAN]")
    print(f"Accuracy (Độ chính xác): {acc*100:.2f}%")
    
    # Ma trận nhầm lẫn
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"True Positives (Đoán trúng Ung thư): {tp}")
    print(f"False Positives (Báo động giả): {fp}")
    print(f"True Negatives (Đoán trúng Lành tính): {tn}")
    print(f"False Negatives (Bỏ sót Ung thư): {fn}")
    
    print(f"\nSensitivity (Độ nhạy - Bắt trúng bệnh): {sensitivity*100:.2f}%")
    print(f"Specificity (Độ đặc hiệu - Không báo động giả): {specificity*100:.2f}%")
    
    # Tính AUC-ROC (chỉ tính được nếu có ít nhất 1 class 0 và 1 class 1)
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"\n>>> AUC-ROC Score: {auc:.4f} <<<")
        if auc > 0.7:
            print("Đánh giá: Mô hình cực kỳ xuất sắc!")
        elif auc > 0.5:
            print("Đánh giá: Mô hình có học được kiến thức, nhưng cần train thêm hoặc cân bằng dữ liệu.")
        else:
            print("Đánh giá: Mô hình dự đoán ngược hoặc chỉ đang đoán lụi.")
    else:
        print("\nKhông thể tính AUC-ROC vì tập dữ liệu test ngẫu nhiên không bốc trúng bệnh nhân Ung thư nào. Hãy chạy lại để lấy mẫu khác.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--csv_path', type=str, required=False, help='Path to NLST CSV')
    parser.add_argument('--is_nlst', action='store_true', help='Use NLST dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--image_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # 1. Khởi tạo Dataset
    dataset = VolumeDataset(data_dir=args.data_dir, csv_file=args.csv_path, is_nlst=args.is_nlst, target_size=(args.image_size, args.image_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2) # Để shuffle=True để bốc ngẫu nhiên cả khỏe và bệnh
    
    # 2. Khởi tạo Mô hình
    model = EndToEndModel().to(device)
    
    # 3. Nạp Checkpoint
    print(f"Đang nạp bộ nhớ từ: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Đã nạp thành công Checkpoint (từ Epoch {checkpoint['epoch']}).")
    
    # 4. Chấm điểm
    evaluate(model, dataloader, device)

if __name__ == '__main__':
    main()
