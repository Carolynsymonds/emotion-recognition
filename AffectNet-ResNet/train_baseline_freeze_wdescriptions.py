
# from data_baseline_freeze import get_data_loaders_clip
from data_raf_db import get_data_loaders_clip
from tqdm import tqdm
from metrics import MetricsLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import clip
import torch.nn as nn
import torch
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from utils import load_config, setup_device, plot_metrics

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

@torch.no_grad()
def build_text_features_mean(by_class_prompts, model, device, emotions, batch_size=64):
    """Averages text embeddings across all templates per class."""
    class_feats = []
    for cls in emotions:
        prompts = by_class_prompts[cls]
        # encode in chunks to avoid OOM
        feats = []
        for i in range(0, len(prompts), batch_size):
            toks = clip.tokenize(prompts[i:i + batch_size]).to(device)
            emb = model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb)
        feats = torch.cat(feats, dim=0).mean(dim=0)  # average over templates
        feats = feats / feats.norm()  # L2 norm
        class_feats.append(feats)
    return torch.stack(class_feats, dim=0)  # [num_classes, d]

@torch.no_grad()
def build_text_features_simple(emotions, model, device):
    """Tokenizes and encodes each emotion label as a CLIP text embedding."""
    tokens = clip.tokenize(emotions).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features  # shape: [num_classes, d]

def by_class_prompt(emotions):
    # Classes must match label IDs 0..7
    templates = [
        "a photo of a {emotion} face",
        "a face showing {emotion}",
        "a person with a {emotion} expression",
        "a close-up of someone looking {emotion}",
        "portrait of a {emotion} person",
    ]

    def a_or_an(word: str) -> str:
        return "an" if word[0].lower() in "aeiou" else "a"

    by_class = {}
    for e in emotions:
        variants = []
        for t in templates:
            phr = t.replace("a {emotion}", f"{a_or_an(e)} {{emotion}}").format(emotion=e)
            variants.append(phr)
        by_class[e] = variants

    return by_class

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

import os
import torch
import matplotlib.pyplot as plt

# CLIP mean/std (OpenAI)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3,1,1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)

emotions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
labels_map_full = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral",
}

def _unnorm_clip(batch):  # batch: [N,C,H,W] or [C,H,W]
    # put mean/std on the same device & dtype as batch
    mean = _CLIP_MEAN.to(device=batch.device, dtype=batch.dtype)
    std  = _CLIP_STD.to(device=batch.device, dtype=batch.dtype)
    return (batch * std + mean).clamp(0, 1)
import os
import torch
import matplotlib.pyplot as plt

# OpenAI CLIP normalization stats
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3,1,1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)

def _unnorm_clip(x):  # x: [N,C,H,W] or [C,H,W]
    mean = _CLIP_MEAN.to(device=x.device, dtype=x.dtype)
    std  = _CLIP_STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0, 1)
def label_to_name(idx, class_names):
    """Support 1-based dicts or 0-based lists/tuples."""
    i = int(idx)
    if isinstance(class_names, dict):     # 1-based mapping
        return class_names.get(i + 1, str(i + 1))
    if isinstance(class_names, (list, tuple)):
        return class_names[i] if 0 <= i < len(class_names) else str(i)
    return str(i)
def visualize_one_per_emotion_baseline(
    val_loader,
    model,
    text_features,
    device,
    class_names=None,          # dict mapping label_id -> name
    alpha=0.45,
    save_dir=None,
    specific_filenames=["test_0002_aligned.jpg", "test_0274_aligned.jpg", "test_0007_aligned.jpg", "test_0003_aligned.jpg","test_0001_aligned.jpg", "test_0017_aligned.jpg", "test_2389_aligned"],   # e.g. ["test_0001_aligned.jpg", ...]
):
    """
    Visualize one sample per class with heatmap overlays.
    Works with RAFDBDatasetCLIP storing samples as (full_path, label) in dataset.samples.
    If 'specific_filenames' is given, picks those; otherwise picks the first occurrence per class.
    """
    model.eval()
    ds = val_loader.dataset

    # ---- Build basename -> index map (robust path matching)
    if not hasattr(ds, "samples"):
        ds = ds.dataset
        # raise RuntimeError("Dataset must expose `samples` with file paths (full_path, label).")

    name2idx = {}
    for i, (full_path, lbl) in enumerate(ds.samples):
        base = os.path.basename(full_path)
        if base not in name2idx:  # keep first occurrence
            name2idx[base] = i

    # ---- Decide which indices to visualize
    indices, labels = [], []
    if specific_filenames:
        for name in specific_filenames:
            idx = name2idx.get(name)
            if idx is None:
                print(f"[WARN] Requested file not found in dataset: {name}")
                continue
            indices.append(idx)
            labels.append(ds.samples[idx][1])
    if not indices:
        print("No samples selected for visualization.")
        return

    # ---- Load images (preprocessed) & labels
    imgs, gts, names = [], [], []
    for idx in indices:
        sample = ds[idx]
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            img_tensor, label, fname = sample
            names.append(fname)
        else:
            img_tensor, label = sample
            names.append(os.path.basename(ds.samples[idx][0]))
        imgs.append(img_tensor.unsqueeze(0))
        gts.append(int(label))

    imgs = torch.cat(imgs, dim=0).to(device)     # [N,C,H,W]
    gts  = torch.tensor(gts, dtype=torch.long, device=device)

    # ---- Compute heatmaps + predictions
    with torch.no_grad():
        heatmaps, pred_cls = token_heatmaps_baseline(model, imgs, text_features)  # [N,Hc,Wc], [N]
        imgs_disp = _unnorm_clip(imgs).detach().cpu()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ---- Visualize/save
    for i in range(imgs_disp.size(0)):
        img_i = imgs_disp[i]                                  # [C,H,W] on CPU
        hm_i  = heatmaps[i].detach().cpu()                    # [Hc,Wc] on CPU
        fig = overlay_heatmap_aligned(
            img_i, hm_i, alpha=alpha, percentile_clip=(60, 99.5), blur=7
        )
        gt   = int(gts[i].item())
        pred = int(pred_cls[i].item()) if torch.is_tensor(pred_cls) else int(pred_cls[i])

        # Safe title lookup even if mapping is mismatched (0-based vs 1-based)
        if class_names:
            gt_name = label_to_name(gt, class_names)  # +1 handled inside
            pred_name = label_to_name(pred, class_names)
            title = f"{names[i]} | gt={gt_name} pred={pred_name}"
        else:
            title = f"{names[i]} | gt={gt} pred={pred}"
        plt.title(title)

        if save_dir:
            fn = f"{os.path.splitext(names[i])[0]}_gt{gt}_pred{pred}.png"
            plt.savefig(os.path.join(save_dir, fn), bbox_inches="tight")
            plt.close(fig)
            print("Saved:", fn)
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
            class_names=labels_map_full,
            alpha=0.45, save_dir=f"{config['checkpoint_parallel']}/vis_once"
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
        metrics_logger.save('metrics_parallel.json')

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
            torch.save(ckpt, f"{config['checkpoint_parallel']}/best_model.pth")
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
    }, f"{config['checkpoint_parallel']}/last.pth")

def plot():
    metrics_logger = MetricsLogger()
    metrics_logger.load("./metrics_parallel.json")

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
    text_features = build_text_features_mean(by_class_prompt(emotions), base_model, device, emotions).detach()
    visualize_one_per_emotion_baseline(
        val_loader, base_model, text_features.to(device), device,
        class_names=emotions,
        alpha=0.45, save_dir=f"{config['checkpoint_parallel']}/vis_once-5"
    )


if __name__ == '__main__':
    # att_heatmap()

    train()
    # plot()
