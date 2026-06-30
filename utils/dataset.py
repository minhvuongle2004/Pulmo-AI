import os
import glob
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class VolumeDataset(Dataset):
    """
    TẬP DỮ LIỆU ĐỌC DICOM 3D
    Nhiệm vụ: Đọc toàn bộ ảnh DICOM của 1 bệnh nhân, sắp xếp theo trục Z, 
    chuyển sang hệ số HU (Hounsfield Unit), và nén thành Tensor 3D.
    """
    def __init__(self, data_dir, csv_file=None, is_nlst=True, target_size=(256, 256)):
        """
        Args:
            data_dir: Đường dẫn đến thư mục chứa dữ liệu (NLST hoặc AAPM).
            csv_file: File CSV chứa nhãn (chỉ dùng cho NLST).
            is_nlst: True nếu đang load bộ NLST (có nhãn Ung thư), False nếu load AAPM.
            target_size: Kích thước [Height, Width] muốn scale ảnh về (default 256x256 để đỡ tốn VRAM).
        """
        self.data_dir = data_dir
        self.is_nlst = is_nlst
        self.target_size = target_size
        
        # Lấy danh sách các thư mục chứa series ảnh của từng bệnh nhân
        # Tạm thời ta lấy tất cả thư mục cấp thấp nhất chứa file .dcm
        self.series_paths = self._find_dicom_series_dirs(data_dir)
        
        if self.is_nlst and csv_file:
            self.labels = self._load_labels(csv_file)
        else:
            self.labels = None

    def _find_dicom_series_dirs(self, root_dir):
        """Hàm đệ quy tìm tất cả các thư mục chứa file DICOM"""
        print(f"Scanning for DICOM data in {root_dir}...")
        series_dirs = set()
        for root, _, files in os.walk(root_dir):
            if any(f.endswith('.dcm') for f in files):
                series_dirs.add(root)
        
        series_dirs = sorted(list(series_dirs))
        print(f"-> Found {len(series_dirs)} series.")
        return series_dirs

    def _load_labels(self, csv_file):
        """Đọc nhãn từ file CSV (chỉ dành cho NLST)"""
        df = pd.read_csv(csv_file)
        # Tạo mapping từ Series UID sang nhãn Cancer (1: Ác tính, 0: Lành tính)
        # Cột 'series' chứa SeriesInstanceUID, cột 'Cancer' chứa nhãn
        labels_dict = dict(zip(df['series'], df['Cancer']))
        print(f"Đã nạp thành công {len(labels_dict)} nhãn từ file CSV.")
        return labels_dict

    def __len__(self):
        return len(self.series_paths)

    def _read_and_sort_dicoms(self, dicom_dir):
        """Đọc và sắp xếp các file DICOM theo tọa độ trục Z (đầu đến chân)"""
        files = glob.glob(os.path.join(dicom_dir, '*.dcm'))
        slices = [pydicom.dcmread(f) for f in files]
        
        # Sắp xếp theo ImagePositionPatient[2] (Trục Z - chiều sâu)
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        # Tính toán khoảng cách giữa các lát cắt (Slice Thickness) nếu cần Resample
        # Ở bước này ta chỉ xếp theo Z
        return slices

    def _get_pixels_hu(self, slices):
        """Chuyển đổi điểm ảnh về hệ số Hounsfield (HU) của cơ thể người"""
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (if safe)
        image = image.astype(np.int16)

        # Trích xuất hệ số để chuyển đổi sang HU
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
                
            image[slice_number] += np.int16(intercept)
        
        return np.array(image, dtype=np.int16)

    def _window_image(self, image, window_center=-600, window_width=1500):
        """Cửa sổ Phổi (Lung Window): Loại bỏ xương và thịt, chỉ giữ lại mô phổi"""
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        windowed_img = np.clip(image, img_min, img_max)
        
        # Chuẩn hóa về [0, 1] cho mạng Nơ-ron dễ học
        normalized_img = (windowed_img - img_min) / window_width
        return normalized_img

    def __getitem__(self, idx):
        # 1. Lấy thư mục DICOM
        dicom_dir = self.series_paths[idx]
        
        # 2. Đọc và sắp xếp
        slices = self._read_and_sort_dicoms(dicom_dir)
        
        # 3. Chuyển sang Hounsfield Units
        hu_volume = self._get_pixels_hu(slices)
        
        # 4. Áp dụng Lung Window và chuẩn hóa về [0, 1]
        lung_volume = self._window_image(hu_volume)
        
        # 5. Đổi shape thành Tensor: [Channel=1, Depth, Height, Width]
        tensor_volume = torch.tensor(lung_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W] để đưa vào interpolate
        
        # 6. Resize về [64, 256, 256] để vừa khít RAM của RTX 3050 và GPU T4 Kaggle
        import torch.nn.functional as F
        target_d, target_h, target_w = 64, self.target_size[0], self.target_size[1]
        resized_volume = F.interpolate(tensor_volume, size=(target_d, target_h, target_w), mode='trilinear', align_corners=False)
        resized_volume = resized_volume.squeeze(0) # Bỏ Batch đi, còn [1, 64, 256, 256]
        
        # 7. Gắn nhãn
        label = 0.0 # Mặc định lành tính
        if self.is_nlst and self.labels:
            # Lấy nhãn thực tế dựa trên Series Instance UID
            series_uid = slices[0].SeriesInstanceUID
            label = float(self.labels.get(series_uid, 0.0))
            
        return resized_volume, torch.tensor([label], dtype=torch.float32)

# --- KHỐI TEST THỬ BẰNG CPU TRÊN MÁY ANH ---
if __name__ == '__main__':
    print("Starting DataLoader Test...")
    # Thử với thư mục AAPM
    test_dir = r"d:\cothuy\3Truc\AAPM-Mayo Clinic\LDCT-and-Projection-data"
    dataset = VolumeDataset(data_dir=test_dir, is_nlst=False)
    
    # Chỉ load 1 batch duy nhất xem có chạy không
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, (volume, label) in enumerate(dataloader):
        print(f"Batch {i}: Volume Shape (1, Depth, Height, Width) = {volume.shape}, Label = {label.item()}")
        break # Chỉ in 1 file rồi dừng để không bị đơ máy
