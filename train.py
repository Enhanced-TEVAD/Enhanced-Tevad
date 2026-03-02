import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
import os 

def sparsity(arr, batch_size, lamda2):
    """Sparsity regularization loss"""
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss

def smooth(arr, lamda1):
    """Temporal smoothness loss"""
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    
    loss = torch.sum((arr2 - arr) ** 2)
    return lamda1 * loss

def l1_penalty(var):
    """L1 penalty for regularization"""
    return torch.mean(torch.norm(var, dim=0))

class SigmoidMAELoss(torch.nn.Module):
    """Sigmoid Mean Absolute Error Loss"""
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)

class SigmoidCrossEntropyLoss(torch.nn.Module):
    """Sigmoid Cross Entropy Loss"""
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(-torch.abs(x))
        return torch.abs(torch.mean(-x * target + torch.clamp(x, min=0) + torch.log(tmp)))

class RTFM_loss(torch.nn.Module):
    """
    RTFM Loss Function with enhanced safety
    """
    def __init__(self, alpha, margin, normal_weight, abnormal_weight):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.normal_weight = normal_weight
        self.abnormal_weight = abnormal_weight
        
        if normal_weight == 1 and abnormal_weight == 1:
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = None

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.cuda()

        # Enhanced safety checks
        if torch.isnan(score).any() or torch.isinf(score).any():
            nan_count = torch.isnan(score).sum().item()
            inf_count = torch.isinf(score).sum().item()
            print(f"⚠️ WARNING: NaN/Inf in scores - NaN: {nan_count}, Inf: {inf_count}")
            
            # If ALL scores are bad, return safe loss
            if torch.isnan(score).all() or torch.isinf(score).all():
                print("❌ CRITICAL: All scores are NaN/Inf! Returning safe loss.")
                return torch.tensor(0.1, requires_grad=True).cuda()
            
            # Replace bad values
            score = torch.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Clamp scores to valid range
        score = torch.clamp(score, min=1e-7, max=1.0 - 1e-7)

        # Binary Cross-Entropy Loss
        if self.criterion:
            loss_cls = self.criterion(score, label)
        else:
            weight = self.abnormal_weight * label + self.normal_weight * (1 - label)
            self.criterion = torch.nn.BCELoss(weight)
            loss_cls = self.criterion(score, label)

        # Check for NaN in BCE loss
        if torch.isnan(loss_cls):
            print("⚠️ WARNING: NaN in BCE loss! Using safe value.")
            loss_cls = torch.tensor(0.1).cuda()

        # Feature Magnitude Loss with safety
        try:
            loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))
            loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)
            loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
        except:
            print("⚠️ WARNING: Feature magnitude loss failed! Using safe value.")
            loss_rtfm = torch.tensor(0.1).cuda()

        # Combined loss
        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total

def train(nloader, aloader, model, args, optimizer, viz, device):
    """
    Training function with enhanced gradient safety
    """
    with torch.set_grad_enabled(True):
        model.train()
        batch_size = args.batch_size
        
        # Get next batch from both loaders
        ninput, ntext, nlabel = next(nloader)
        ainput, atext, alabel = next(aloader)
      
        # Concatenate normal and abnormal videos
        input = torch.cat((ninput, ainput), 0).to(device)
        text = torch.cat((ntext, atext), 0).to(device)

        # Forward pass
        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(input, text)

        # Reshape scores for loss computation
        scores = scores.view(batch_size * 32 * 2, -1)
        scores = scores.squeeze()
        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        # Compute losses
        loss_criterion = RTFM_loss(args.alpha, 100, args.normal_weight, args.abnormal_weight)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        
        if args.extra_loss:
            cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal,
                                  feat_select_abn) + loss_smooth + loss_sparse
        else:
            cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)

        # CRITICAL: Check for NaN before backward
        if torch.isnan(cost) or torch.isinf(cost):
            print("⚠️ CRITICAL: NaN/Inf in total loss! Skipping batch.")
            optimizer.zero_grad()
            return

        # Backward pass with enhanced safety
        optimizer.zero_grad()
        cost.backward()
        
        # AGGRESSIVE gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check and handle NaN gradients
        nan_detected = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"❌ NaN/Inf gradient in {name} - Zeroing")
                    param.grad.data.zero_()
                    nan_detected = True
        
        if nan_detected:
            print("⚠️ NaN gradients detected and zeroed")
        
        # Only update if we have valid gradients
        has_valid_grads = any(torch.isfinite(p.grad).any() for p in model.parameters() if p.grad is not None)
        
        if has_valid_grads:
            optimizer.step()
        else:
            print("❌ No valid gradients - skipping update")
        
        # Log losses
        viz.plot_lines('loss', cost.item())
        viz.plot_lines('smooth loss', loss_smooth.item())
        viz.plot_lines('sparsity loss', loss_sparse.item())
