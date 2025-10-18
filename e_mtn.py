import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

class PDCBlock(nn.Module):
    """Pyramid Dilated Convolution block with multi-scale 1D convolutions."""
    def __init__(self, in_channels, out_channels):
        super(PDCBlock, self).__init__()
        # Three parallel convolutions with different dilation rates
        self.conv_d1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv_d3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        
        # Add BatchNorm for stability (CRITICAL FIX)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # After concatenation, fuse the channels back to out_channels
        self.fuse = nn.Conv1d(3*out_channels, out_channels, kernel_size=1)
        self.bn_fuse = nn.BatchNorm1d(out_channels)
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch_init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

    def forward(self, x):
        # Apply parallel dilated convolutions with BatchNorm + ReLU
        out1 = F.relu(self.bn1(self.conv_d1(x)))
        out2 = F.relu(self.bn2(self.conv_d2(x)))
        out3 = F.relu(self.bn3(self.conv_d3(x)))
        
        # Concatenate along the channel dimension
        out = torch.cat([out1, out2, out3], dim=1)
        
        # Fuse concatenated features
        out = F.relu(self.bn_fuse(self.fuse(out)))
        return out


class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel-wise feature recalibration."""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        # Ensure reduction doesn't make dimension too small
        self.reduction = min(reduction, channels // 2)
        reduced_channels = max(channels // self.reduction, 1)
        
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        
        # Initialize weights
        torch_init.xavier_uniform_(self.fc1.weight)
        torch_init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        b, c, t = x.size()
        
        # Squeeze: average over temporal dimension
        y = x.mean(dim=2)  # (batch, channels)
        
        # Excitation with proper scaling
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))  # (batch, channels), each in [0,1]
        
        # Scale is in [0,1], preventing extreme amplification
        y = y.view(b, c, 1)
        return x * y


class TransformerEncoderBlock(nn.Module):
    """A single Transformer Encoder layer with stability improvements."""
    def __init__(self, channels, nhead=8, dim_feedforward=1024, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # Ensure channels is divisible by nhead
        if channels % nhead != 0:
            nhead = self._find_valid_nhead(channels, nhead)
            print(f"⚠️ Adjusted nhead to {nhead} to be divisible by channels={channels}")
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,  # Add dropout for stability
            batch_first=False,
            norm_first=True  # CRITICAL: Pre-normalization is more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def _find_valid_nhead(self, channels, preferred_nhead):
        """Find the largest valid nhead <= preferred_nhead that divides channels."""
        for h in range(preferred_nhead, 0, -1):
            if channels % h == 0:
                return h
        return 1

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = x.permute(2, 0, 1)  # (seq_len, batch, channels)
        
        # Pass through transformer with proper shape
        out = self.transformer(x)  # (seq_len, batch, channels)
        
        # Permute back
        out = out.permute(1, 2, 0)  # (batch, channels, seq_len)
        return out
