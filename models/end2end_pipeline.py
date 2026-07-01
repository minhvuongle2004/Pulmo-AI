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
        
        from torch.utils.checkpoint import checkpoint
        
        # 1. TẦNG 1: Xử lý DenoMamba cho toàn bộ lát cắt
        # Sử dụng Gradient Checkpointing để giải phóng VRAM (giảm 80% bộ nhớ)
        # Chỉ tính 1 lần mỗi lát cắt thay vì 3 lần như code cũ
        denoised_slices = []
        for z in range(D):
            current_slice = volume[:, :, z, :, :]
            # checkpoint function, args..., use_reentrant=False
            denoised = checkpoint(self.denomamba, current_slice, use_reentrant=False)
            denoised_slices.append(denoised)
            
        cleaned_slices = []
        
        # 2. TẦNG 2: Xử lý CoreDiff kết hợp giải phẫu 3D
        for z in range(D):
            current_slice = volume[:, :, z, :, :]
            denoised_slice = denoised_slices[z]
            
            # Xử lý biên
            if z == 0:
                prev_slice = denoised_slice
            else:
                # Không được gán tensor có gradient qua lại phức tạp nếu dùng checkpoint
                # Dùng chính denoised của z-1 để đơn giản hóa đồ thị tính toán
                prev_slice = denoised_slices[z-1] 
                
            if z == D - 1:
                next_slice = denoised_slice
            else:
                next_slice = denoised_slices[z+1]
                
            # Ghép 3 lát cắt lại thành input 3 channels: [B, 3, H, W]
            context_3d = torch.cat([prev_slice, denoised_slice, next_slice], dim=1)
            
            if t_diffusion is None:
                t_diffusion = torch.zeros(B, device=volume.device, dtype=torch.long)
                
            # Sử dụng Checkpoint cho CoreDiff
            # CoreDiff forward args: (x, time, y, x_end)
            diffused_slice = checkpoint(self.corediff, context_3d, t_diffusion, current_slice, current_slice, use_reentrant=False)
            cleaned_slices.append(diffused_slice)
            
        # 3. CẦU NỐI (Bridge)
        clean_volume = torch.stack(cleaned_slices, dim=2)
        
        # 4. TẦNG 3
        # SybilNet không cần checkpoint vì nó nhận nguyên cục 3D và kiến trúc ResNet3D đã tối ưu
        sybil_output = self.sybil(clean_volume)
        
        return sybil_output
