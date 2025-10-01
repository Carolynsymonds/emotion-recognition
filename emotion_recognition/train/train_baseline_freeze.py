
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_affectnet import get_data_loaders_clip
from tqdm import tqdm
from emotion_recognition.metrics import MetricsLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import clip
import torch.nn as nn
import torch
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from emotion_recognition.utils import load_config, setup_device, plot_metrics
from emotion_recognition.models import build_text_features_simple, build_text_features_mean, by_class_prompt

def update_ema(val, ema, alpha=0.1):
    return val if ema is None else (alpha * val + (1 - alpha) * ema)

def encode_images_trainable(model, images):
    feats = model.encode_image(images)                # grads flow into last block
    feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 norm
    return feats

def train_epoch_last_block(train_loader, model, optimizer, criterion, device, text_features, epoch):
    """
    Train only the last transformer block of CLIP's visual encoder,
    using fixed text_features for zero-shot classification.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    ema_loss, ema_acc = None, None

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    sample_logged = False

    scale = model.logit_scale.exp()
    print("scale", scale)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        img_feats = encode_images_trainable(model, images)          # [B, D]
        logits = scale * (img_feats @ text_features.t())
        loss     = criterion(logits, labels)

        if not torch.isfinite(logits).all():
            print("Non-finite logits! stats:",
                  logits.min().item(), logits.max().item(), logits.mean().item(), logits.std().item())
            print("Any NaN in img_feats:", torch.isnan(img_feats).any().item())
            print("Any NaN in text_features:", torch.isnan(text_features).any().item())
            optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.isfinite(loss):
            print("Non-finite loss!", loss.item())
            print("labels min/max:", labels.min().item(), labels.max().item(), labels.dtype)
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        # THIS IS FOR DEBUG
        with torch.no_grad():
            last_block = model.visual.transformer.resblocks[-1]
            grad_norm_last = sum(p.grad.norm().item() for p in last_block.parameters() if p.grad is not None)
            frozen_block = model.visual.transformer.resblocks[-2]
            grad_norm_frozen = sum(
                (p.grad.norm().item() if p.grad is not None else 0) for p in frozen_block.parameters())

        if not sample_logged:
            print(f"Grad Norm - Last Block: {grad_norm_last:.6f} | Frozen Block: {grad_norm_frozen:.6f}")
        # THIS IS FOR DEBUG

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        batch_acc = (preds == labels).float().mean().item()
        ema_loss = update_ema(loss.item(), ema_loss)
        ema_acc  = update_ema(batch_acc, ema_acc)

        with torch.no_grad():
            logit_mean = logits.mean().item()
            logit_std  = logits.std().item()

        if not sample_logged:
            print("Label:      ", labels.tolist())
            print("Prediction: ", preds.tolist())
            top3 = torch.topk(logits, k=3, dim=1)
            for i in range(min(5, images.size(0))):  # print 5 samples
                print(
                    f"Label: {labels[i].item()}, Top-3: {top3.indices[i].tolist()}, Scores: {top3.values[i].tolist()}")
            sample_logged = True

        progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'ema_loss': f'{ema_loss:.3f}',
            'ema_acc': f'{ema_acc:.2f}',
            'lr': optimizer.param_groups[0]['lr'],
            'logμ': f'{logit_mean:.2f}',
            'logσ': f'{logit_std:.2f}',
        })

    epoch_loss = running_loss / max(1, total)
    accuracy   = correct / max(1, total)
    return epoch_loss, accuracy

def mean_class_accuracy(per_class_acc):
    per_class_acc = np.asarray(per_class_acc, dtype=float)
    # ignore NaNs if any class has 0 support
    return np.nanmean(per_class_acc)

@torch.no_grad()
def evaluate_last_block(val_loader, model, device, criterion, num_classes, text_features):
    """
    Evaluation for last-block fine-tuning setup.
    Uses fixed text_features for classification.
    """
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        scale = model.logit_scale.exp()
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            img_feats = encode_images_trainable(model, images)        # last block trainable
            logits    = scale * (img_feats @ text_features.t()   )              # cosine similarity

            loss = criterion(logits, labels)
            val_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_loss /= max(1, total)
    val_acc   = correct / max(1, total)

    # Macro-F1: ignore absent classes by default; avoid warnings
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Confusion matrix (rows = true class)
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    support = cm.sum(axis=1)  # samples per (true) class
    tp = np.diag(cm).astype(float)

    # Per-class accuracy with safe division (0 if support==0)
    per_class_acc = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)

    # ---- MCA that ignores empty classes ----
    valid_mask = (support > 0)

    # Optional: specifically drop Contempt (idx=7) only if it's empty
    if num_classes > 7 and support[7] == 0:
        valid_mask[7] = False

    mca = float(per_class_acc[valid_mask].mean()) if valid_mask.any() else 0.0

    return val_loss, val_acc, macro_f1, cm, per_class_acc, mca, support

# Function moved to models.py

# Function moved to models.py

# Function moved to models.py

import math, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def extract_patch_tokens(clip_model, images):
    """
    Extract patch tokens (including CLS token) from the visual transformer.

    Args:
        clip_model: The loaded CLIP model with ViT visual backbone.
        images: Preprocessed images tensor [B, 3, H, W] on same device as model.

    Returns:
        tokens: Tensor of shape [B, N+1, hidden_dim] containing CLS + patch tokens.
    """
    v = clip_model.visual

    # Move images type to match model weights
    x = images.to(dtype=v.conv1.weight.dtype)

    # Patch embedding with conv1
    x = v.conv1(x)  # Shape: [B, 768, H/16, W/16]

    # Flatten spatial dimensions to tokens sequence
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, N, 768]

    # Add class token to start of tokens sequence
    cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)  # [B,1,768]
    x = torch.cat([cls, x], dim=1)  # [B, N+1, 768]

    # Add positional embeddings
    x = x + v.positional_embedding.to(x.dtype)

    # Apply pre-transformer LayerNorm
    x = v.ln_pre(x)

    # Pass through transformer blocks (sequencing of residual attention blocks)
    x = v.transformer(x.permute(1, 0, 2))  # Permute to [seq_len, batch, dim]
    x = x.permute(1, 0, 2)  # Back to [B, seq_len, dim]

    # Now x contains patch tokens and CLS token embeddings
    return x
def project_visual_tokens(clip_model, tokens_768):
    x = clip_model.visual.ln_post(tokens_768)
    if clip_model.visual.proj is not None:
        x = x @ clip_model.visual.proj
    return x

@torch.no_grad()
def token_heatmaps_baseline(model, images, text_features):
    """
    images: [B,3,H,W] (CLIP-preprocessed)
    text_features: [C,512], L2-normalized class prototypes (detached)
    returns:
      up_hmaps: [B,H,W] heatmaps in [0,1], upsampled to image size
      pred_cls: [B] predicted class id (same path as training)
    """
    model.eval()
    B, _, H, W = images.shape
    t = F.normalize(text_features, dim=-1)                     # [C,512]
    scale = model.logit_scale.exp()

    # ---- Pred class exactly like training (encode_image path) ----
    img_feats = model.encode_image(images)                     # [B,512]
    img_feats = F.normalize(img_feats, dim=-1)
    logits_cls = scale * (img_feats @ t.t())                   # [B,C]
    pred_cls = logits_cls.argmax(dim=1)                        # [B]

    # ---- Patch tokens for localization ----
    tokens_768 = extract_patch_tokens(model, images)           # [B,N+1,768]
    tokens_512 = project_visual_tokens(model, tokens_768)      # [B,N+1,512]

    v_patches = F.normalize(tokens_512[:, 1:, :], dim=-1)      # [B,N,512]

    # Patch→class logits & probs
    logits_loc = scale * (v_patches @ t.t().unsqueeze(0))      # [B,N,C]
    probs_loc  = torch.softmax(logits_loc, dim=-1)             # [B,N,C]

    # Select predicted class channel → per-patch score
    idx = pred_cls.view(-1, 1, 1).expand(-1, probs_loc.size(1), 1)  # [B,N,1]
    heat = torch.gather(probs_loc, 2, idx).squeeze(-1)         # [B,N]

    # Reshape N → S×S (e.g., 14×14 for ViT-B/16 with 224x224)
    N = heat.size(1)
    S = int(round(math.sqrt(N)))
    assert S * S == N, f"Patch count {N} is not a square"
    heat = heat.view(B, 1, S, S)

    # Normalize each heatmap to [0,1]
    hmin = heat.amin(dim=(2,3), keepdim=True)
    hmax = heat.amax(dim=(2,3), keepdim=True)
    heat = (heat - hmin) / (hmax - hmin + 1e-8)                # [B,1,S,S]

    # Upsample to image size for overlay
    up_hmaps = F.interpolate(heat, size=(H, W), mode="bilinear", align_corners=False)
    up_hmaps = up_hmaps.squeeze(1)                              # [B,H,W]
    return up_hmaps, pred_cls


def overlay_heatmap(img_01, heatmap_01, alpha=0.45):
    """
    img_01: [3,H,W] in [0,1] (unnormalized for display)
    heatmap_01: [H,W] in [0,1]
    returns: matplotlib Figure
    """
    fig = plt.figure(figsize=(4,4))
    plt.imshow(img_01.permute(1,2,0).cpu().numpy())
    plt.imshow(heatmap_01.cpu().numpy(), cmap="jet", alpha=alpha)
    plt.axis("off")
    return fig

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def overlay_heatmap_aligned(img_CHW: torch.Tensor,
                            cam_HW: torch.Tensor,
                            alpha: float = 0.45,
                            percentile_clip=(60, 99.5),
                            blur=None):
    """
    img_CHW: float tensor [C,H,W] in [0,1] (already unnormalized for display)
    cam_HW : float tensor [H',W'] CAM (unnormalized). Will be resized to HxW.
    alpha  : overlay strength
    percentile_clip: (lo, hi) for robust per-image CAM normalization
    blur: optional int Gaussian kernel size (odd) for mild smoothing, or None
    Returns: matplotlib Figure
    """
    assert img_CHW.ndim == 3, "img must be [C,H,W]"
    C, H, W = img_CHW.shape

    # --- Normalize CAM with robust clipping
    cam = cam_HW.detach().float().cpu().numpy()
    lo, hi = np.percentile(cam, percentile_clip)
    cam = np.clip((cam - lo) / (hi - lo + 1e-6), 0, 1)

    # --- Resize CAM to image size
    # Use matplotlib's imshow resizing by drawing at full size; or do explicit resize:
    # do explicit resize with numpy + PIL to keep deps minimal
    try:
        from PIL import Image
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((W, H), resample=Image.BILINEAR)
        cam = np.asarray(cam_img).astype(np.float32) / 255.0
    except ImportError:
        # Fallback: simple np.kron (nearest). (Should almost never hit.)
        ry, rx = H / cam.shape[0], W / cam.shape[1]
        cam = np.kron(cam, np.ones((int(np.ceil(ry)), int(np.ceil(rx)))))[0:H, 0:W]

    # --- Optional light blur for nicer blobs
    if blur and blur > 1 and blur % 2 == 1:
        try:
            from scipy.ndimage import gaussian_filter
            cam = gaussian_filter(cam, sigma=blur/6.0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        except Exception:
            pass  # okay to skip if scipy not present

    # --- Colorize CAM (JET) and alpha blend over RGB image
    jet = cm.get_cmap('jet')
    heat_rgba = jet(cam)      # [H,W,4], floats in [0,1]
    heat_rgb = heat_rgba[..., :3]

    img = img_CHW.permute(1, 2, 0).detach().cpu().numpy()  # [H,W,C], [0,1]
    if C == 1:
        img = np.repeat(img, 3, axis=2)

    overlay = alpha * heat_rgb + (1 - alpha) * img
    overlay = np.clip(overlay, 0, 1)

    # --- Plot nicely
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis('off')
    return fig

@torch.no_grad()
def visualize_one_per_emotion_baseline(val_loader, model, text_features, device,
                                       class_names=None, alpha=0.45, save_dir=None):
    model.eval()
    seen = set()
    keep_imgs, keep_labels = [], []

    # references ["7.jpg (5)", "16.jpg (1)", "17.jpg (4)", "36.jpg(3)", "3.jpg (0)", "19.jpg (2)", "47.jpg (6)"]
    target_names = ["7.jpg", "218.jpg", "17.jpg", "36.jpg", "3.jpg", "19.jpg", "1008.jpg"]
    from torch.utils.data import Subset

    keep_imgs, keep_labels = [], []
    ds = val_loader.dataset
    is_subset = isinstance(ds, Subset)
    base_ds = ds.dataset if is_subset else ds
    subset_idxs = ds.indices if is_subset else None

    # Get (path, label) list from the BASE dataset
    items = getattr(base_ds, "samples", getattr(base_ds, "imgs", None))
    if items is None:
        raise AttributeError(
            "Dataset exposes neither .samples nor .imgs; add a path list to your Dataset."
        )
    import os
    # Map basename -> list of base indices (handle duplicates)
    name_to_base = {}
    for bi, (p, lbl) in enumerate(items):
        name = os.path.basename(p)
        name_to_base.setdefault(name, []).append(bi)

    # If Subset, build reverse map: base index -> subset index
    base_to_subset = {b: s for s, b in enumerate(subset_idxs)} if is_subset else None

    # CLIP normalization (only for display reverse)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def unnorm(x, mean=CLIP_MEAN, std=CLIP_STD):
        # x: [C,H,W] tensor on CPU
        if x.dim() != 3:
            return x
        m = torch.tensor(mean).view(3, 1, 1)
        s = torch.tensor(std).view(3, 1, 1)
        if x.size(0) == 3:
            return (x * s + m).clamp(0, 1)
        return x.clamp(0, 1)

    keep_imgs, keep_labels = [], []
    missing = []

    for target_file in target_names:
        print("looking for:", target_file)
        base_idxs = name_to_base.get(target_file, [])
        if not base_idxs:
            missing.append(target_file)
            continue

        # If duplicates exist, we’ll take all that are actually in this split
        found_any = False
        for bi in base_idxs:
            if is_subset:
                si = base_to_subset.get(bi)
                if si is None:
                    # present in base, but not part of this validation subset
                    continue
                img_tensor, label = ds[si]
            else:
                img_tensor, label = base_ds[bi]

            # Make batch dim and move to device for later model use
            img_b = img_tensor.unsqueeze(0).to(device)
            keep_imgs.append(img_b)
            keep_labels.append(label)

            # Quick visualization (optional)
            t_disp = unnorm(img_tensor.detach().cpu())
            plt.figure(figsize=(4, 4))
            if t_disp.size(0) == 1:
                plt.imshow(t_disp.squeeze(0), cmap="gray")
            else:
                plt.imshow(t_disp.permute(1, 2, 0))
            plt.axis("off")
            plt.title(f"{target_file} (label={label})")
            plt.show()

            found_any = True

        if not found_any:
            # File exists in base dataset but not in this subset split
            missing.append(target_file + " (not in this split)")

    if missing:
        raise FileNotFoundError(f"Not found in this dataset/split: {missing}")

    print(f"Collected {len(keep_imgs)} images.")

    # Collect one per class
    # for images, labels in val_loader:
    #     images, labels = images.to(device), labels.to(device)
    #     for i in range(images.size(0)):
    #         c = labels[i].item()
    #         if c not in seen:
    #             seen.add(c)
    #             keep_imgs.append(images[i:i+1])  # keep batch dim
    #             keep_labels.append(c)
    #     if len(seen) == text_features.size(0):
    #         break

    if not keep_imgs:
        print("No samples found for visualization.")
        return

    # Stack kept images -> [N, C, H, W] (still model-preprocessed)
    imgs = torch.cat(keep_imgs, dim=0)  # [N, C, H, W]

    # --- Compute heatmaps with your method (must return [N, Hc, Wc]) + preds
    heatmaps, pred_cls = token_heatmaps_baseline(model, imgs, text_features)  # user-provided

    # --- Unnormalize to [0,1] for display (adjust to your preprocessing!)
    # If your CLIP preprocessing used mean=0.5, std=0.5:
    def unnorm(x):  # x: [N,C,H,W]
        return (x * 0.5 + 0.5).clamp(0, 1)

    imgs_disp = unnorm(imgs)

    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- Show/save each overlay with aligned CAM
    for i in range(imgs_disp.size(0)):
        fig = overlay_heatmap_aligned(
            imgs_disp[i],                  # [C,H,W] in [0,1]
            heatmaps[i],                   # [Hc,Wc] (unnormalized CAM)
            alpha=alpha,
            percentile_clip=(60, 99.5),
            blur=7                         # try 5 or 7 for look like your target example
        )
        gt = keep_labels[i]
        pred = pred_cls[i].item()
        title = f"gt={gt} pred={pred}"
        if class_names:
            title = f"gt={class_names[gt]} pred={class_names[pred]}"
        plt.title(title)

        if save_dir:
            fn = f"class_{gt}_pred_{pred}.png"
            plt.savefig(os.path.join(save_dir, fn), bbox_inches="tight")
            plt.close(fig)
            print("Saved image:", fn)
        else:
            plt.show()


import math
def train():
    # -- Setup --
    config = load_config('config.yaml')
    device = setup_device()

    print(f"Training for {config['num_epochs']} epochs")
    print(f"Using {config['batch_size']} mini-batch size")
    print(f"Using {config['learning_rate']} learning rate")


    # 1) -- Data --
    train_loader, val_loader, test_loader = get_data_loaders_clip(config, device)

    # 2) -- Model: freeze CLIP completely --
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.float()  # force full FP32
    for p in model.parameters():
        p.data = p.data.float()
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
    model.logit_scale.requires_grad = False  # keep frozen
    # Unfreeze only the last transformer block of the visual encoder
    for p in model.visual.transformer.resblocks[-1].parameters():
        p.requires_grad = True

    # Set fixed temperature τ = 0.07
    tau = 0.07
    with torch.no_grad():
        model.logit_scale.fill_(math.log(1.0 / tau))

    # Freeze so it doesn't get updated
    model.logit_scale.requires_grad_(False)

    # ONLY USED FOR EVAL
    emotions = ["neutral", "happy", "sad", "surprised", "fearful", "disgusted", "angry"]
    # text_features = build_text_features_simple(emotions, model, device)
    text_features = build_text_features_mean(by_class_prompt(emotions), model, device, emotions).detach()

    optimizer = torch.optim.AdamW(
        list(model.visual.transformer.resblocks[-1].parameters()),
        lr=config["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # small smoothing helps on noisy AffectNet
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    model.to(device)

    # 4- Initialize metrics
    metrics_logger = MetricsLogger()


    # THIS IS FOR DEBUG
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable_params:,} | Frozen params: {frozen_params:,}")
    #THIS IS FOR DEBUG

    # 5 - Training loop
    best_val_loss = float('inf')
    patience = 5
    early_stopping_counter = 0
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")

        # if (epoch + 1) % 2 == 0:  # e.g., every 2 epochs
        visualize_one_per_emotion_baseline(
            val_loader, model, text_features.to(device), device,
            class_names=["neutral", "happy", "sad", "surprised", "fearful", "disgusted", "angry"],
            alpha=0.45, save_dir=f"{config['checkpoint_dir']}/vis_once"
        )

        train_loss, train_accuracy = train_epoch_last_block(train_loader, model, optimizer, criterion, device, text_features, epoch)
        val_loss, val_acc, val_f1, val_cm, val_pc, val_mca, support = evaluate_last_block(val_loader, model, device, criterion, config['num_classes'], text_features)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}:")
        print(f"Training   - Loss: {train_loss:.4f} | Acc: {train_accuracy:.2%}")

        print(f"Validation - Acc: {val_acc:.2%} | Macro-F1: {val_f1:.3f} | Mean Class Acc: {val_mca:.2%}")
        print("Per-class acc:", " ".join(f"{i}:{a:.2%}" for i, a in enumerate(val_pc)))
        print("Support:", support.tolist())

        # Log metrics
        metrics_logger.log_epoch(
            train_loss=train_loss,
            val_loss=val_loss,
            accuracy=train_accuracy,
            val_accuracy=val_acc,
            macro_f1=val_f1,
            mean_class_accuracy=val_mca,
            per_class_accuracy=[float(x) for x in val_pc]
        )
        metrics_logger.save('metrics.json')

        print(f'best val loss: {best_val_loss:.4f}')
        print(f'val loss: {val_loss:.4f}')

        # Save best model (by val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),  # CLIP (frozen)
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                "val_macro_f1": float(val_f1),
                "val_mean_class_acc": float(val_mca),
                "per_class_acc": [float(x) for x in val_pc],
                'config': config,
            }
            torch.save(ckpt, f"{config['checkpoint_dir']}/best_model.pth")
            print('Saved Best Model!')
        else:
            early_stopping_counter += 1
            print(f" No improvement. Early stopping counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print(" Early stopping triggered.")
                break
    # (Optional) also save last checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': float(val_acc),
        'val_macro_f1': float(val_f1),
        'val_mean_class_acc': float(val_mca),
        'per_class_acc': [float(x) for x in val_pc],
        'config': config,
    }, f"{config['checkpoint_dir']}/last.pth")

def plot():
    metrics_logger = MetricsLogger()
    metrics_logger.load("./metrics.json")

    plot_metrics(metrics_logger.get_metrics_history(), "./checkpoints")


def att_heatmap():
    global device, p
    config = load_config('config.yaml')
    device = setup_device()
    train_loader, val_loader, test_loader = get_data_loaders_clip(config, device)
    import torch
    ckpt_path = "./history-3/baseline-no-contempt/best_model_no_contempt.pth"  # or config['checkpoint_dir']/best_model.pth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) Recreate the model architecture exactly as during training
    #    (use your model class + args; you can read ckpt['config'] if helpful)
    # 2) -- Model: freeze CLIP completely --
    base_model, preprocess = clip.load("ViT-B/16", device=device)
    base_model.float()  # force full FP32
    for p in base_model.parameters():
        p.data = p.data.float()
    # Freeze all
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.logit_scale.requires_grad = False  # keep frozen
    # Unfreeze only the last transformer block of the visual encoder
    for p in base_model.visual.transformer.resblocks[-1].parameters():
        p.requires_grad = True
    # Set fixed temperature τ = 0.07
    tau = 0.07
    with torch.no_grad():
        base_model.logit_scale.fill_(math.log(1.0 / tau))
    # Freeze so it doesn't get updated
    base_model.logit_scale.requires_grad_(False)
    # 2) Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"]

    # 3) Strip common prefixes so keys match CLIP's names
    def strip_prefix(d, prefix):
        return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}

    # Try multiple mappings to fit whatever was saved
    candidates = []
    # (a) raw
    candidates.append(state)
    # (b) remove DataParallel 'module.'
    if any(k.startswith("module.") for k in state):
        candidates.append({k.replace("module.", "", 1): v for k, v in state.items()})
    # (c) if wrapped like 'clip.xxx'
    if any(k.startswith("clip.") for k in state):
        s = strip_prefix(state, "clip.")
        candidates.append(s)
        if any(k.startswith("module.clip.") for k in state):
            s = strip_prefix({k.replace("module.", "", 1): v for k, v in state.items()}, "clip.")
            candidates.append(s)
    loaded = False
    for cand in candidates:
        try:
            missing, unexpected = base_model.load_state_dict(cand, strict=False)
            # Heuristic: accept if most params matched
            if len(missing) + len(unexpected) < 10:
                print("Loaded with minor mismatches.")
                if missing:   print("Missing keys:", missing[:10], "...")
                if unexpected: print("Unexpected keys:", unexpected[:10], "...")
                loaded = True
                break
        except Exception as e:
            # Try next mapping
            pass
    if not loaded:
        # Last attempt: just report what failed with the raw dict
        missing, unexpected = base_model.load_state_dict(state, strict=False)
        print("WARNING: could not confidently align keys.")
        print("Missing keys:", missing[:20])
        print("Unexpected keys:", unexpected[:20])
    base_model.to(device).eval()
    print("Loaded epoch:", ckpt.get("epoch"))
    print("Val loss/acc:", ckpt.get("val_loss"), ckpt.get("val_acc"))
    emotions = ["neutral", "happy", "sad", "surprised", "fearful", "disgusted", "angry"]
    text_features = build_text_features_mean(by_class_prompt(emotions), base_model, device, emotions).detach()
    visualize_one_per_emotion_baseline(
        val_loader, base_model, text_features.to(device), device,
        class_names=["neutral", "happy", "sad", "surprised", "fearful", "disgusted", "angry"],
        alpha=0.45, save_dir=f"{config['checkpoint_dir']}/vis_once-5"
    )


if __name__ == '__main__':
    att_heatmap()

    # train()
    # plot()
