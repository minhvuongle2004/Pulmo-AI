import torch
import torch.nn as nn

# Tích hợp 3 khối mô hình
from .denomamba_arch import DenoMamba
from .corediff.corediff_wrapper import Network as CoreDiff
from .sybil.sybil import SybilNet

class EndToEndModel(nn.Module):
    """
    TRÁI TIM CỦA DỰ ÁN - MÔ HÌNH END-TO-END 3 TRỤC
    
    Luồng dữ liệu (Data Flow):
    1. Đầu vào (Input): Khối LDCT 3D kích thước [Batch, 1, Depth, Height, Width]
    2. Tầng 1 (DenoMamba): Lặp qua từng lát cắt 2D theo trục Z (Depth), lọc nhiễu không gian.
    3. Tầng 2 (CoreDiff): Nhóm 3 lát cắt 2D liên tiếp lại (z-1, z, z+1) để khuếch tán 
       và khôi phục tính liên tục của cấu trúc giải phẫu 3D.
    4. Cầu nối (Bridge): Sử dụng torch.stack để xếp các lát cắt đã làm sạch thành khối 3D nguyên vẹn.
    5. Tầng 3 (Sybil): Đưa toàn bộ khối 3D sạch vào phân tích rủi ro ung thư.
    """
    def __init__(self, corediff_context=True, sybil_dropout=0.0):
        super(EndToEndModel, self).__init__()
        
        # --- Khởi tạo Tầng 1: DenoMamba ---
        # Input: 1 channel (Grayscale 2D CT slice)
        self.denomamba = DenoMamba(inp_channels=1, out_channels=1)
        
        # --- Khởi tạo Tầng 2: CoreDiff ---
        # Input: 3 channels (3 lát cắt kề nhau)
        self.corediff = CoreDiff(in_channels=3, out_channels=1, context=corediff_context)
        
        # --- Khởi tạo Tầng 3: Sybil ---
        self.sybil = SybilNet(dropout=sybil_dropout)
        
    def forward(self, volume, t_diffusion=None):
        """
        Quá trình truyền tới (Forward pass) kết hợp 3 mô hình.
        Args:
            volume: Tensor 3D, shape [B, 1, D, H, W]
            t_diffusion: Time step cho CoreDiff (nếu cần huấn luyện Diffusion từ đầu)
        """
        B, C, D, H, W = volume.shape
        assert C == 1, "Input volume must have exactly 1 channel"
        
        cleaned_slices = []
        
        # 1. & 2. ĐI QUA TẦNG 1 VÀ TẦNG 2 (Xử lý 2D)
        # Chúng ta sẽ lặp qua từng lát cắt theo trục chiều sâu (D)
        for z in range(D):
            # Lấy lát cắt hiện tại: [B, 1, H, W]
            current_slice = volume[:, :, z, :, :]
            
            # --- TẦNG 1 ---
            denoised_slice = self.denomamba(current_slice)
            
            # --- TẦNG 2 ---
            # Để CoreDiff hoạt động (khôi phục giải phẫu 3D), nó cần ngữ cảnh từ lát trước và sau
            # Xử lý biên (Padding cho lát đầu và cuối)
            if z == 0:
                prev_slice = denoised_slice # Fake lát trước bằng chính nó
            else:
                prev_slice = cleaned_slices[-1] # Lấy kết quả đã làm sạch của lát liền trước
                
            if z == D - 1:
                next_slice = self.denomamba(volume[:, :, z, :, :]) # Fake lát sau
            else:
                next_slice = self.denomamba(volume[:, :, z+1, :, :]) # Tính trước Tầng 1 cho lát tiếp theo
                
            # Ghép 3 lát cắt lại thành input 3 channels: [B, 3, H, W]
            context_3d = torch.cat([prev_slice, denoised_slice, next_slice], dim=1)
            
            # Tạm thời fix cứng time step (t) bằng 0 cho quá trình inference/fine-tuning (tuỳ chiến lược)
            if t_diffusion is None:
                t_diffusion = torch.zeros(B, device=volume.device, dtype=torch.long)
                
            # Tạo các tham số adjust giả lập cho corediff (y, x_end) - Ở thực tế cần trích xuất đặc trưng
            # Ở đây em truyền chính ảnh vào để giữ nguyên cấu trúc hàm forward gốc của CoreDiff
            diffused_slice = self.corediff(context_3d, t_diffusion, y=current_slice, x_end=current_slice)
            
            cleaned_slices.append(diffused_slice)
            
        # 3. CẦU NỐI (Bridge)
        # Ghép các lát cắt 2D đã sạch lại thành một khối 3D: [B, 1, D, H, W]
        # Torch.stack ở dim=2 (Depth) sẽ giữ cho Gradient (đạo hàm) không bị đứt gãy
        clean_volume = torch.stack(cleaned_slices, dim=2)
        
        # 4. TẦNG 3
        # Đưa khối 3D sạch vào Sybil để phân loại
        sybil_output = self.sybil(clean_volume)
        
        return sybil_output
