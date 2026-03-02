import matplotlib.pyplot as plt
import torch, sys
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
from utils import get_gt

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        
        for i, (input, text) in enumerate(dataloader):  # test set has videos
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            text = text.to(device)
            text = text.permute(0, 2, 1, 3)
            
            # Forward pass
            outputs = model(input, text)
            (
                score_abnormal, score_normal,
                feat_select_abn, feat_select_normal,
                feat_abn_bottom, feat_select_normal_bottom,
                logits, scores_nor_bottom,
                scores_nor_abn_bag, feat_magnitudes
            ) = outputs
            
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            
            # Debug for NaNs
            if torch.isnan(sig).any():
                print(f"⚠️ NaN detected at batch {i}! Replacing with zeros.")
                sig = torch.nan_to_num(sig, nan=0.0, posinf=1.0, neginf=0.0)
            
            pred = torch.cat((pred, sig))
        
        # Ground truth
        gt = get_gt(args.dataset, args.gt)
        
        # Convert predictions to NumPy
        pred = pred.cpu().detach().numpy()
        pred = np.repeat(pred, 16)  # same score for each frame in the clip
        
        # ✅ Check for invalid values before metrics
        if not np.all(np.isfinite(pred)):
            print("❌ Warning: NaN or Inf detected in predictions! Fixing them...")
            print("  Before:", pred[:10])
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            print("  After:", pred[:10])
        
        # 🔥 CRITICAL FIX: Handle length mismatch
        gt_len = len(gt)
        pred_len = len(pred)
        
        if pred_len != gt_len:
            print(f"⚠️ Length mismatch detected: GT={gt_len}, Pred={pred_len}, Diff={abs(gt_len-pred_len)}")
            
            if pred_len > gt_len:
                # Truncate predictions to match GT
                print(f"   Truncating predictions from {pred_len} to {gt_len}")
                pred = pred[:gt_len]
            else:
                # Pad predictions to match GT (repeat last value)
                print(f"   Padding predictions from {pred_len} to {gt_len}")
                padding_needed = gt_len - pred_len
                last_value = pred[-1] if len(pred) > 0 else 0.5
                pred = np.concatenate([pred, np.full(padding_needed, last_value)])
        
        # Double check lengths match
        assert len(pred) == len(gt), f"❌ Length mismatch after fix: pred={len(pred)}, gt={len(gt)}"
        
        # Compute metrics safely
        try:
            fpr, tpr, threshold = roc_curve(list(gt), pred)
            precision, recall, th = precision_recall_curve(list(gt), pred)
            pr_auc = auc(recall, precision)
            rec_auc = auc(fpr, tpr)
            ap = average_precision_score(list(gt), pred)
        except Exception as e:
            print(f"❌ Error computing metrics: {e}")
            print(f"   GT shape: {np.array(gt).shape}, Pred shape: {pred.shape}")
            print(f"   GT range: [{np.min(gt)}, {np.max(gt)}]")
            print(f"   Pred range: [{np.min(pred):.4f}, {np.max(pred):.4f}]")
            raise
        
        print(f"AP  : {ap:.4f}")
        print(f"AUC : {rec_auc:.4f}")
        
        # Visualization logging
        if viz is not None:
            viz.plot_lines('pr_auc', pr_auc)
            viz.plot_lines('auc', rec_auc)
            viz.lines('scores', pred)
            viz.lines('roc', tpr, fpr)
        
        if args.save_test_results:
            import os
            save_dir = 'output_metrics'
            os.makedirs(save_dir, exist_ok=True)
            
            np.save(f'{save_dir}/{args.dataset}_pred.npy', pred)
            np.save(f'{save_dir}/{args.dataset}_fpr.npy', fpr)
            np.save(f'{save_dir}/{args.dataset}_tpr.npy', tpr)
            np.save(f'{save_dir}/{args.dataset}_precision.npy', precision)
            np.save(f'{save_dir}/{args.dataset}_recall.npy', recall)
            np.save(f'{save_dir}/{args.dataset}_auc.npy', rec_auc)
            np.save(f'{save_dir}/{args.dataset}_ap.npy', ap)
        
        return rec_auc, ap
