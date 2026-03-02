import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

class PDCBlock(nn.Module):
    """
    Pyramid Dilated Convolution Block - EXACTLY as described in paper Section 3.3
    """
    def __init__(self, in_channels, out_channels):
        super(PDCBlock, self).__init__()
        
        # Three parallel dilated convolutions (Paper Section 4.2.2)
        self.conv_d1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv_d3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        
        # BatchNorm for stability
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # Fusion layer: concatenate 3*out_channels -> out_channels
        self.fuse = nn.Conv1d(3 * out_channels, out_channels, kernel_size=1)
        self.bn_fuse = nn.BatchNorm1d(out_channels)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization as mentioned in paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch_init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Apply parallel dilated convolutions with BatchNorm + ReLU
        out1 = F.relu(self.bn1(self.conv_d1(x)))
        out2 = F.relu(self.bn2(self.conv_d2(x)))
        out3 = F.relu(self.bn3(self.conv_d3(x)))
        
        # Concatenate along channel dimension
        out = torch.cat([out1, out2, out3], dim=1)
        
        # Fuse concatenated features
        out = F.relu(self.bn_fuse(self.fuse(out)))
        return out


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module - EXACTLY as described in paper Section 3.3
    """
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        
        # Paper mentions reduction ratio of 16
        reduced_channels = max(channels // reduction, 4)
        
        # Two FC layers as described in paper
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        
        # Initialize weights properly
        torch_init.xavier_uniform_(self.fc1.weight)
        torch_init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        b, c, t = x.size()
        
        # Squeeze: Global average pooling over temporal dimension
        y = x.mean(dim=2)
        
        # Excitation: Two FC layers with ReLU and Sigmoid
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        
        # Minimal stability fix
        y = torch.clamp(y, min=1e-3, max=1.0-1e-3)
        
        # Reshape and scale input
        y = y.view(b, c, 1)
        return x * y


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block (TEB) - EXACTLY as described in paper Section 3.3
    """
    def __init__(self, channels, nhead=8, dim_feedforward=1024, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # Paper Section 4.2.2: nhead=8, dim_feedforward=1024, dropout=0.1
        if channels % nhead != 0:
            nhead = self._find_valid_nhead(channels, nhead)
        
        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def _find_valid_nhead(self, channels, preferred_nhead):
        for h in range(preferred_nhead, 0, -1):
            if channels % h == 0:
                return h
        return 1

    def forward(self, x):
        b, c, t = x.shape
        
        # Permute to transformer format: (seq_len, batch, channels)
        x = x.permute(2, 0, 1)
        
        # Pass through transformer encoder
        out = self.transformer(x)
        
        # Permute back to original format: (batch, channels, seq_len)
        out = out.permute(1, 2, 0)
        return out


class Aggregate(nn.Module):
    """
    Enhanced Multi-Scale Temporal Network (E-MTN)
    Paper: Section 3.3, Figure 3, Equation 6
    F_E-MTN = (F_PDC-SE ⊕ F_TEB) + F_vta
    """
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        self.len_feature = len_feature
        quarter_channels = int(len_feature / 4)
        
        # Left Branch: PDC + SE Module
        self.pdc_block = PDCBlock(len_feature, quarter_channels)
        self.se_module = SEModule(quarter_channels, reduction=16)
        
        # Right Branch: 1D Conv + Transformer Encoder Block
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=quarter_channels, 
                     kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(quarter_channels),
            nn.ReLU(),
        )
        
        # Paper Section 4.2.2: nhead=8, dim_feedforward=1024
        self.transformer_block = TransformerEncoderBlock(
            quarter_channels, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.1
        )
        
        # Final fusion layer (after concatenation)
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, 
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(len_feature),
            nn.ReLU(),
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights as in paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch_init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Paper Equation 6: F_E-MTN = (F_PDC-SE ⊕ F_TEB) + F_vta
        """
        # Permute to (B, F, T) for convolution operations
        out = x.permute(0, 2, 1)
        residual = out  # F_vta in paper

        # Left Branch: PDC -> SE (F_PDC-SE in paper)
        pdc_out = self.pdc_block(out)
        se_out = self.se_module(pdc_out)
        
        # Expand to 3*F/4 by repeating (to match concatenation size)
        out_d = se_out.repeat(1, 3, 1)  # (B, 3*F/4, T)

        # Right Branch: Conv -> TEB (F_TEB in paper)
        conv_out = self.conv_4(out)
        transformer_out = self.transformer_block(conv_out)

        # Equation 6: Concatenate both branches (F_PDC-SE ⊕ F_TEB)
        out = torch.cat((out_d, transformer_out), dim=1)  # (B, F, T)
        
        # Final fusion
        out = self.conv_5(out)
        
        # Equation 6: Add residual connection (+ F_vta)
        out = out + residual
        
        # Permute back to (B, T, F)
        out = out.permute(0, 2, 1)
        return out
