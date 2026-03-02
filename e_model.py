import torch
import torch.nn as nn
import torch.nn.init as torch_init
from e_mtn import PDCBlock, SEModule, TransformerEncoderBlock, Aggregate

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    """Weight initialization for model layers"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    """Non-Local Block (kept for compatibility, not used in Enhanced MTN)"""
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    """1D Non-Local Block (kept for compatibility)"""
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class Model(nn.Module):
    """
    Main Model for Video Anomaly Detection
    Uses SINGLE Enhanced MTN for fused features (as in paper "V + A + T" approach)
    """
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.fusion = args.fusion
        self.batch_size = args.batch_size
        self.feature_group = args.feature_group
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        # Calculate fused feature size based on fusion method
        if self.feature_group == 'both':
            if args.fusion == 'concat':
                # Concatenate visual (1024) + text (768) = 1792 features
                self.fused_feature_size = args.feature_size + args.emb_dim
            elif args.fusion == 'add' or args.fusion == 'product':
                # Transform to same dimension then fuse
                self.fused_feature_size = args.emb_dim
                self.fc_vis = nn.Linear(args.feature_size, args.emb_dim)
                self.fc_text = nn.Linear(args.emb_dim, args.emb_dim)
            elif 'up' in args.fusion:
                # Project to higher dimension
                self.fused_feature_size = args.feature_size + args.emb_dim
                self.fc_vis = nn.Linear(args.feature_size, self.fused_feature_size)
                self.fc_text = nn.Linear(args.emb_dim, self.fused_feature_size)
        elif self.feature_group == 'text':
            self.fused_feature_size = args.emb_dim
        else:
            self.fused_feature_size = args.feature_size
        
        # SINGLE Enhanced MTN for fused features (Paper "V + A + T" approach)
        self.Aggregate = Aggregate(len_feature=self.fused_feature_size)
        
        # Classification layers
        self.fc1 = nn.Linear(self.fused_feature_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs, text):
        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs  # shape=[batch_size*2, 10, 32, feature_size]
        bs, ncrops, t, f = out.size()
        bs2, ncrops2, t2, f2 = text.size()

        out = out.view(-1, t, f)
        out2 = text.view(-1, t2, f2)

        # FUSE FEATURES FIRST (as in paper "V + A + T" approach)
        if self.feature_group == 'both':
            if self.fusion == 'concat':
                # Paper Equation 4: Concatenation method
                out = torch.cat([out, out2], dim=2)  # (B, T, F_v + F_t)
            elif self.fusion == 'product':
                vis_proj = self.relu(self.fc_vis(out))
                text_proj = self.relu(self.fc_text(out2))
                out = vis_proj * text_proj
            elif self.fusion == 'add':
                vis_proj = self.relu(self.fc_vis(out))
                text_proj = self.relu(self.fc_text(out2))
                out = vis_proj + text_proj
            elif self.fusion == 'add_up':
                vis_proj = self.relu(self.fc_vis(out))
                text_proj = self.relu(self.fc_text(out2))
                out = vis_proj + text_proj
        elif self.feature_group == 'text':
            out = out2
        # else: use visual features only (out remains unchanged)

        # Apply SINGLE Enhanced MTN to fused features
        out = self.Aggregate(out)
        out = self.drop_out(out)

        features = out
        
        # Classification head
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)

        # Split into normal and abnormal
        normal_features = features[0:self.batch_size * ncrops]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size * ncrops:]
        abnormal_scores = scores[self.batch_size:]

        # Calculate feature magnitudes
        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]
        afea_magnitudes = feat_magnitudes[self.batch_size:]
        n_size = nfea_magnitudes.shape[0]

        # Handle inference case
        if nfea_magnitudes.shape[0] == 1:
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        # Process abnormal videos -> select top-k features
        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, -1)
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)

        total_select_abn_feature = torch.zeros(0).cuda()
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)

        # Process normal videos -> select top-k features
        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, -1)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0).cuda()
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes
