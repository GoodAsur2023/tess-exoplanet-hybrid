"""
model.py — Dual-View CNN-Transformer for exoplanet transit classification.

"""
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, channels=[32, 64, 128], kernel_size=5, dropout=0.0):
        super().__init__()
        layers = []
        in_c = in_channels
        for out_c in channels:
            layers.append(nn.Conv1d(in_c, out_c, kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout))
            in_c = out_c
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DualViewTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        m_cfg = config["model"]
        self.d_model = m_cfg["d_model"]
        self.fusion_type = m_cfg.get("fusion_type", "meta_token")
        
        self.global_cnn = FeatureExtractor(dropout=m_cfg["cnn_dropout"])
        self.local_cnn = FeatureExtractor(dropout=m_cfg["cnn_dropout"])
        cnn_out_ch = m_cfg.get("cnn_channels", [32, 64, 128])[-1]
        
        self.global_proj = nn.Linear(cnn_out_ch, self.d_model)
        self.local_proj = nn.Linear(cnn_out_ch, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=m_cfg["nhead"], 
            dim_feedforward=m_cfg["dim_feedforward"], dropout=m_cfg["transformer_dropout"], batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=m_cfg["num_transformer_layers"])
        
        if self.fusion_type == "mlp":
            self.meta_mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
            self.classifier = nn.Sequential(nn.Linear(self.d_model + 16, 64), nn.ReLU(), nn.Dropout(m_cfg["classifier_dropout"]), nn.Linear(64, 1))
            
        elif self.fusion_type == "meta_token":
            self.meta_proj = nn.Linear(3, self.d_model) 
            self.classifier = nn.Sequential(nn.Linear(self.d_model, 64), nn.ReLU(), nn.Dropout(m_cfg["classifier_dropout"]), nn.Linear(64, 1))
            
        elif self.fusion_type == "film":
            self.film_gen = nn.Linear(3, 2 * self.d_model) 
            self.classifier = nn.Sequential(nn.Linear(self.d_model, 64), nn.ReLU(), nn.Dropout(m_cfg["classifier_dropout"]), nn.Linear(64, 1))
            
        elif self.fusion_type == "astrophysics":
            self.astro_mlp = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
            self.classifier = nn.Sequential(nn.Linear(self.d_model + 16, 64), nn.ReLU(), nn.Dropout(m_cfg["classifier_dropout"]), nn.Linear(64, 1))
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

    def forward(self, x_global, x_local, x_meta):
        g_feat = torch.mean(self.global_cnn(x_global), dim=-1)
        l_feat = torch.mean(self.local_cnn(x_local), dim=-1)
        g_embed = self.global_proj(g_feat) 
        l_embed = self.local_proj(l_feat)  
        
        if self.fusion_type == "mlp":
            seq = torch.stack([g_embed, l_embed], dim=1)
            trans_out = torch.mean(self.transformer(seq), dim=1)
            meta_out = self.meta_mlp(x_meta)
            fused = torch.cat([trans_out, meta_out], dim=1)
            return self.classifier(fused)
            
        elif self.fusion_type == "meta_token":
            meta_token = self.meta_proj(x_meta)
            seq = torch.stack([meta_token, g_embed, l_embed], dim=1)
            trans_out = self.transformer(seq)
            return self.classifier(torch.mean(trans_out, dim=1))
            
        elif self.fusion_type == "film":
            film_params = self.film_gen(x_meta)
            gamma, beta = film_params.chunk(2, dim=-1) 
            g_embed = g_embed * (1 + gamma) + beta
            l_embed = l_embed * (1 + gamma) + beta
            seq = torch.stack([g_embed, l_embed], dim=1)
            trans_out = torch.mean(self.transformer(seq), dim=1)
            return self.classifier(trans_out)
            
        elif self.fusion_type == "astrophysics":
            L_norm = (x_meta[:, 2]**2) * (x_meta[:, 0]**4)
            M_norm = x_meta[:, 1] * (x_meta[:, 2]**2)
            physics_vec = torch.cat([x_meta, L_norm.unsqueeze(1), M_norm.unsqueeze(1)], dim=1)
            
            seq = torch.stack([g_embed, l_embed], dim=1)
            trans_out = torch.mean(self.transformer(seq), dim=1)
            astro_out = self.astro_mlp(physics_vec)
            fused = torch.cat([trans_out, astro_out], dim=1)
            return self.classifier(fused)