"""
model.py — Dual-View CNN-Transformer for exoplanet transit classification.

Architecture:
┌─────────────────────────────────────────────────────────┐
│  Global view (2048)          Local view (201)                              │
│       ↓                            ↓                                       │
│  GlobalCNNEncoder            LocalCNNEncoder                               │
│       ↓                            ↓                                       │
│      Linear projection to d_model (both streams)                           │
│       ↓                            ↓                                       │
│       Positional encoding (sinusoidal)                                     │
│     └──────────┬──────────────┘                                   │
│     Concatenate along sequence dim                                         │
│                    ↓                                                       │
│ Transformer Encoder (cross-view self-attention)                            │
│                    ↓                                                       │
│ Global Average Pooling → Dropout → Linear                                  │
│                    ↓                                                       │
│            p(planet) ∈ [0, 1]                                              |
└─────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


# ── Positional Encoding ───────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── CNN Encoders ──────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_length: int,
        channels: list[int],
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            stride = 2 if i > 0 else 1
            padding = kernel_size // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        T = in_length
        for i in range(len(channels)):
            stride = 2 if i > 0 else 1
            padding = kernel_size // 2
            T = (T + 2 * padding - kernel_size) // stride + 1
        self.out_length = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)       # (B, 1, L)
        x = self.conv(x)          # (B, C, T')
        return x.transpose(1, 2)  # (B, T', C)


# ── Main Model ────────────────────────────────────────────────────────────────
class DualViewTransformer(nn.Module):
    def __init__(
        self,
        global_length: int = 2048,
        local_length: int = 201,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 5,
        cnn_dropout: float = 0.1,
        d_model: int = 256,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 512,
        transformer_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        self.global_cnn = CNNEncoder(global_length, cnn_channels, cnn_kernel_size, cnn_dropout)
        self.local_cnn  = CNNEncoder(local_length,  cnn_channels, cnn_kernel_size, cnn_dropout)
        cnn_out_ch = cnn_channels[-1]

        self.global_proj = nn.Linear(cnn_out_ch, d_model)
        self.local_proj  = nn.Linear(cnn_out_ch, d_model)

        combined_len = self.global_cnn.out_length + self.local_cnn.out_length
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=combined_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.global_cnn(global_view)   # (B, T_g, C)
        l = self.local_cnn(local_view)     # (B, T_l, C)
        g = self.global_proj(g)            # (B, T_g, d_model)
        l = self.local_proj(l)             # (B, T_l, d_model)
        x = torch.cat([g, l], dim=1)       # (B, T_g+T_l, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)            # (B, T, d_model)
        pooled = x.mean(dim=1)             # (B, d_model)
        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        return logits, x


# ── Baseline (1D-CNN only) ────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    def __init__(
        self,
        global_length: int = 2048,
        local_length: int = 201,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 5,
        cnn_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        self.global_cnn = CNNEncoder(global_length, cnn_channels, cnn_kernel_size, cnn_dropout)
        self.local_cnn  = CNNEncoder(local_length,  cnn_channels, cnn_kernel_size, cnn_dropout)
        g_flat = self.global_cnn.out_length * cnn_channels[-1]
        l_flat = self.local_cnn.out_length  * cnn_channels[-1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(g_flat + l_flat, 512),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        g = self.global_cnn(global_view).flatten(1)
        l = self.local_cnn(local_view).flatten(1)
        x = torch.cat([g, l], dim=-1)
        logits = self.head(x).squeeze(-1)
        return logits, None


# ── Factory ───────────────────────────────────────────────────────────────────
def build_model(config: dict, device: torch.device) -> DualViewTransformer:
    cfg = config["model"]
    cfg_data = config["data"]
    model = DualViewTransformer(
        global_length=cfg_data["global_view_length"],
        local_length=cfg_data["local_view_length"],
        cnn_channels=cfg["cnn_channels"],
        cnn_kernel_size=cfg["cnn_kernel_size"],
        cnn_dropout=cfg["cnn_dropout"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_transformer_layers=cfg["num_transformer_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        transformer_dropout=cfg["transformer_dropout"],
        classifier_dropout=cfg["classifier_dropout"],
    )
    return model.to(device)


def build_baseline(config: dict, device: torch.device) -> BaselineCNN:
    cfg = config["model"]
    cfg_data = config["data"]
    model = BaselineCNN(
        global_length=cfg_data["global_view_length"],
        local_length=cfg_data["local_view_length"],
        cnn_channels=cfg["cnn_channels"],
        cnn_kernel_size=cfg["cnn_kernel_size"],
        cnn_dropout=cfg["cnn_dropout"],
        classifier_dropout=cfg["classifier_dropout"],
    )
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cpu")
    model = DualViewTransformer()
    print(f"DualViewTransformer parameters: {count_parameters(model):,}")
    gv = torch.randn(4, 2048)
    lv = torch.randn(4, 201)
    logits, tokens = model(gv, lv)
    print(f"Output logits shape : {logits.shape}")
    print(f"Token embeddings shape: {tokens.shape}")