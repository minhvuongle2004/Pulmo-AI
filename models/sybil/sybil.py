# TẦNG 3: Sybil (Ung thư phổi - Dự đoán rủi ro)
# Chức năng: Nhận đầu vào là một khối 3D Tensor (sau khi đã được Tầng 1 và Tầng 2 làm sạch)
# Khối 3D này sẽ đi qua 3D ResNet (image_encoder) để trích xuất đặc trưng.
# Lớp phân loại cuối cùng được sửa lại để dự đoán trực tiếp xác suất Ác tính (Binary Classification)
import torch
import torch.nn as nn
import torchvision
from .pooling_layer import MultiAttentionPool

class SybilNet(nn.Module):
    def __init__(self, dropout=0.0):
        super(SybilNet, self).__init__()

        self.hidden_dim = 512

        # 3D ResNet-18 pretrained để trích xuất đặc trưng hình khối 3D
        encoder = torchvision.models.video.r3d_18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])

        self.pool = MultiAttentionPool()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout)

        # [SỬA ĐỔI LỚN]: Thay thế Cumulative_Probability_Layer bằng một lớp Linear đơn giản
        # Mục đích: Phù hợp với nhãn dữ liệu (0: Lành tính, 1: Ác tính) của file CSV
        self.classifier = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        # x: Tensor 3D đầu vào có kích thước (Batch, Channel, Depth, Height, Width)
        output = {}
        x = self.image_encoder(x)
        pool_output = self.aggregate_and_classify(x)
        output["activ"] = x
        output.update(pool_output)
        
        # Sigmoid để ép giá trị về khoảng [0, 1] (Xác suất ung thư)
        output["prob"] = torch.sigmoid(pool_output["logit"])

        return output

    def aggregate_and_classify(self, x):
        pool_output = self.pool(x)

        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        
        # Áp dụng lớp Linear để phân loại
        pool_output["logit"] = self.classifier(pool_output["hidden"])

        return pool_output

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location="cpu")
        args = checkpoint["args"]
        model = SybilNet(args)

        # Remove 'model' from param names
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)  # type: ignore
        return model

