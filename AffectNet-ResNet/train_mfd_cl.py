
from data_features import get_data_loaders_clip
from tqdm import tqdm
from metrics import MetricsLogger
import clip
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from utils import load_config, setup_device, plot_metrics
import torch.nn as nn
import torch
import torch.nn.functional as F

def update_ema(val, ema, alpha=0.1):
    return val if ema is None else (alpha * val + (1 - alpha) * ema)

def encode_images_trainable(model, images):
    feats = model.encode_image(images)                # grads flow into last block
    feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 norm
    return feats


def compute_affinities(v_cls, v_patches, text_features, temp):
    # Normalize embeddings
    v_cls = F.normalize(v_cls, dim=-1)          # [B, D]
    v_patches = F.normalize(v_patches, dim=-1)  # [B, N, D]
    text_features = F.normalize(text_features, dim=-1)  # [C, D]

    # Global affinities: [B, C]
    logits_global = torch.matmul(v_cls, text_features.t()) / temp

    # Local affinities: [B, N, C]
    logits_local = torch.matmul(v_patches, text_features.t()) / temp

    # Temperature scaling and softmax along class dimension
    affinity_global = torch.softmax(logits_global, dim=-1)  # [B, C]
    affinity_local = torch.softmax(logits_local, dim=-1)    # [B, N, C]

    return affinity_global, affinity_local

def topk_local_affinity(affinity_local, k=16):
    # affinity_local: [B, N, C], topk over N patches
    topk_values, _ = torch.topk(affinity_local, k=k, dim=1)  # [B, k, C]
    affinity_local_decoupled = topk_values.mean(dim=1)       # mean over top-K, shape [B, C]
    return affinity_local_decoupled
def combine_affinities(affinity_global, affinity_local_decoupled, gamma=0.3):
    combined_affinity = gamma * affinity_global + (1 - gamma) * affinity_local_decoupled  # [B, C]
    return combined_affinity

def mfd_fused_logits_from_features(v_cls, v_patches, text_features, scale, k=16, gamma=0.3):
    """
    v_cls:     [B, D]    (CLS after adapter; D=512 in your setup)
    v_patches: [B, N, D] (patch tokens after adapter)
    text_features: [C, D] (L2-normalized)
    scale: CLIP logit scale, i.e., model.logit_scale.exp()
    returns: fused logits [B, C] suitable for CrossEntropyLoss
    """
    # L2-normalize visual features (text already normalized)
    v_cls     = F.normalize(v_cls, dim=-1)
    v_patches = F.normalize(v_patches, dim=-1)
    t         = F.normalize(text_features, dim=-1)

    # Global logits (standard CLIP): scale * cosine
    logits_cls = scale * (v_cls @ t.t())                                # [B, C]

    # Local logits per patch
    logits_loc = scale * (v_patches @ t.t().unsqueeze(0))               # [B, N, C]

    # Per-patch class probs, then Top-K over patches per class
    A_loc = torch.softmax(logits_loc, dim=-1)        # [B, N, C]
    A_loc_bcn = A_loc.permute(0, 2, 1)               # [B, C, N]
    k_eff = min(k, A_loc_bcn.shape[-1])
    topk_vals = torch.topk(A_loc_bcn, k=k_eff, dim=2).values            # [B, C, k]
    A_loc_mean = topk_vals.mean(dim=2).clamp_min(1e-8)                  # [B, C]

    # Fuse in logit space (numerically stable): logits + log(prob)
    fused_logits = gamma * logits_cls + (1.0 - gamma) * torch.log(A_loc_mean)
    return fused_logits

def project_visual_tokens(clip_model, tokens_768):
    x = clip_model.visual.ln_post(tokens_768)
    if clip_model.visual.proj is not None:
        x = x @ clip_model.visual.proj
    return x

@torch.no_grad()
def encode_text_batch(model, desc_tokens):
    # desc_tokens: LongTensor [B, 77] from clip.tokenize
    txt = model.encode_text(desc_tokens)
    return F.normalize(txt, dim=-1)

def clip_itc_loss(img_emb, txt_emb, temp=0.07):
    # img_emb, txt_emb: [B, D], both L2-normalized
    logits_i2t = img_emb @ txt_emb.t() / temp
    logits_t2i = txt_emb @ img_emb.t() / temp
    targets = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i2t = F.cross_entropy(logits_i2t, targets)
    loss_t2i = F.cross_entropy(logits_t2i, targets)
    return 0.5 * (loss_i2t + loss_t2i)
import math
def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            if torch.isfinite(g).all():
                total += g.norm(2).item() ** 2
    return math.sqrt(total)

def tensor_stats(x, name, k=3):
    x_det = x.detach()
    finite = torch.isfinite(x_det)
    return {
        f'{name}_min': float(x_det[finite].min().item()) if finite.any() else float('nan'),
        f'{name}_max': float(x_det[finite].max().item()) if finite.any() else float('nan'),
        f'{name}_mean': float(x_det[finite].mean().item()) if finite.any() else float('nan'),
    }
import torch
import torch.nn.functional as F

class MultiPositiveContrastiveLoss(torch.nn.Module):
    """
    Multi-positive CLIP-style contrastive loss.

    Args:
        temperature (float): softmax temperature (tau). We scale logits by 1/tau.
        symmetric (bool): if True, compute image->text and text->image losses and average.
        pos_weight (float or None): optional BCE positive weight to counter class imbalance.

    Forward inputs:
        img_embeds: (B, D) L2-normalized or will be normalized here
        txt_embeds: (M, D) where M = sum_i P_i (total captions/text for the batch images)
        pos_index:  list/tuple length B. pos_index[i] is a 1D LongTensor of indices into txt_embeds
                    indicating which texts belong to image i (can be variable #positives per image).
    """
    def __init__(self, temperature=0.07, symmetric=True, pos_weight=None):
        super().__init__()
        self.tau = temperature
        self.symmetric = symmetric
        self.register_buffer("pos_weight_tensor",
                             torch.tensor(pos_weight) if pos_weight is not None else None)

    def forward(self, img_embeds, txt_embeds, pos_index):
        # Normalize (safe if already normalized)
        img = F.normalize(img_embeds, dim=-1)
        txt = F.normalize(txt_embeds, dim=-1)

        # Similarity matrix S = img @ txt^T -> (B, M)
        logits = (img @ txt.t()) / self.tau   # (B, M)

        # Build multi-hot target matrix Y of shape (B, M)
        B, M = logits.shape
        device = logits.device
        targets = torch.zeros((B, M), dtype=torch.float32, device=device)
        for i, idx in enumerate(pos_index):
            targets[i, idx.to(device)] = 1.0

        # BCE-with-logits row-wise: image->text
        # Optionally weight positives to balance (far fewer positives than negatives)
        if self.pos_weight_tensor is not None:
            pos_w = self.pos_weight_tensor.to(device)
            loss_i2t = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_w, reduction="mean"
            )
        else:
            loss_i2t = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

        if not self.symmetric:
            return loss_i2t

        # Symmetric term: text->image
        # Build targets^T : (M, B). We need mapping from each text to its owning image.
        # Derive inverse mapping from pos_index
        # For each image i, for each text idx in pos_index[i], text2img[idx] = i
        text2img = torch.full((M,), -1, dtype=torch.long, device=device)
        for i, idx in enumerate(pos_index):
            text2img[idx.to(device)] = i
        assert (text2img >= 0).all(), "Every text must map to exactly one image"

        # logits_t2i: (M, B) = txt @ img^T / tau (or just logits.T)
        logits_t2i = logits.t()  # (M, B)

        # targets_t2i: one-hot over images for each text
        targets_t2i = F.one_hot(text2img, num_classes=B).to(torch.float32)  # (M, B)

        if self.pos_weight_tensor is not None:
            # Here, positives per row = 1; pos_weight still ok to keep symmetry
            loss_t2i = F.binary_cross_entropy_with_logits(
                logits_t2i, targets_t2i, pos_weight=self.pos_weight_tensor.to(device), reduction="mean"
            )
        else:
            loss_t2i = F.binary_cross_entropy_with_logits(
                logits_t2i, targets_t2i, reduction="mean"
            )

        return 0.5 * (loss_i2t + loss_t2i)

def train_epoch_adapter_only(train_loader, model, optimizer, criterion, device, text_features, epoch, adapter, scheduler, gamma, k,  alpha=0.2, beta=0.1, itc_temp=0.07, k_itc=None):
    """
     Train the Emotion-aware Adapter on top of frozen CLIP visual encoder.
    Use fixed text_features for classification.
    """
    model.eval()  # Freeze CLIP backbone in eval mode
    adapter.train()  # Train adapter

    running_loss, correct, total = 0.0, 0, 0
    ema_loss, ema_acc = None, None

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    sample_logged = False
    scale2 = model.logit_scale.exp()
    mp_criterion = MultiPositiveContrastiveLoss(temperature=0.07, symmetric=True, pos_weight=20.0)

    for images, labels, per_image_text in progress_bar:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            patch_tokens_768 = extract_patch_tokens(model, images)  # [B, N+1, 768]
            tokens_512 = project_visual_tokens(model, patch_tokens_768).to(device)  # [B, N+1, 512]

        tokens = adapter(tokens_512)  # [B, N+1, 512]
        v_cls = tokens[:, 0, :]  # [B, 512]
        v_patches = tokens[:, 1:, :]  # [B, N, 512]

        scale = model.logit_scale.exp()
        logits = mfd_fused_logits_from_features(v_cls, v_patches, text_features, scale, k=k, gamma=gamma)
        loss_ce = criterion(logits, labels)

        # ----- per-image multi-positive text embeddings (ITC) -----
        # per_image_text can be: List[List[str]] (preferred) or List[str]
        all_texts = []
        pos_index = []  # list of LongTensors; pos_index[i] -> indices in all_texts that belong to image i
        offset = 0
        for t in per_image_text:
            texts_i = t if isinstance(t, (list, tuple)) else [t]
            if not texts_i:
                texts_i = ["a face with an unspecified expression"]
            n_i = len(texts_i)
            print(f'texts {n_i}')
            all_texts.extend(texts_i)
            pos_index.append(torch.arange(offset, offset + n_i, dtype=torch.long, device=device))
            offset += n_i

        # Tokenize & encode all captions for the batch
        desc_tokens = clip.tokenize(all_texts).to(device, non_blocking=True)
        with torch.no_grad():
            txt_embeds = encode_text_batch(model, desc_tokens)  # (M, 512), L2-normalized inside encode_text_batch

        # Compute multi-positive contrastive loss (image->text and text->image)
        # mp_criterion = MultiPositiveContrastiveLoss(temperature=itc_temp, symmetric=True, pos_weight=pos_w)
        loss_itc_global = mp_criterion(v_cls, txt_embeds, pos_index)

        # ----- total loss -----
        loss = loss_ce + alpha * loss_itc_global

        # ===== anomaly checks (fail fast) =====
        if not torch.isfinite(loss):
            print("[ANOMALY] Non-finite loss detected!")
            print({**tensor_stats(logits, "logits"), **tensor_stats(v_cls, "v_cls")})
            print("Labels sample:", labels[:8].tolist())
            raise RuntimeError("Loss is NaN/Inf")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)  # Grad clipping on adapter params
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # <-- per batch, no args

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

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        batch_acc = (preds == labels).float().mean().item()
        ema_loss = update_ema(loss.item(), ema_loss)
        ema_acc  = update_ema(batch_acc, ema_acc)

        if not sample_logged:
            print("Sample labels:", labels[:5].tolist())
            print("Sample predictions:", preds[:5].tolist())
            top3 = torch.topk(logits, k=3, dim=1)
            for i in range(min(5, images.size(0))):
                print(
                    f"Label: {labels[i].item()}, Top-3 classes: {top3.indices[i].tolist()}, Scores: {top3.values[i].tolist()}")
            sample_logged = True

        progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'ema_loss': f'{ema_loss:.3f}',
            'ema_acc': f'{ema_acc:.2f}',
            'lr': optimizer.param_groups[0]['lr']
        })

    epoch_loss = running_loss / max(1, total)
    accuracy   = correct / max(1, total)
    return epoch_loss, accuracy

def mean_class_accuracy(per_class_acc):
    per_class_acc = np.asarray(per_class_acc, dtype=float)
    # ignore NaNs if any class has 0 support
    return np.nanmean(per_class_acc)

@torch.no_grad()
def evaluate_last_block(val_loader, model, device, criterion, num_classes, text_features, epoch, adapter, gamma, k):
    """
    Evaluation for last-block fine-tuning setup.
    Uses fixed text_features for classification.
    """
    model.eval()
    adapter.eval()

    assert text_features.size(0) == num_classes, \
        f"num_classes ({num_classes}) must match text_features.size(0) ({text_features.size(0)})"
    text_features = text_features.to(device)

    val_loss = 0.0
    correct = total = 0
    all_preds, all_labels = [], []

    scale = model.logit_scale.exp()  # scalar tensor

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}")

    for images, labels, per_image_text in progress_bar:

        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)

        tokens_768 = extract_patch_tokens(model, images)  # [B, N+1, 768]
        tokens_512 = project_visual_tokens(model, tokens_768)  # [B, N+1, 512]

        tokens = adapter(tokens_512)  # [B, N+1, 512]
        v_cls, v_patches = tokens[:, 0, :], tokens[:, 1:, :]  # [B, 512], [B, N, 512]

        # Fused logits (global + local top-K) against class prototypes only
        logits = mfd_fused_logits_from_features(
            v_cls, v_patches, text_features, scale, k=k, gamma=gamma
        )
        loss = criterion(logits, labels)

        val_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_loss /= max(1, total)
    val_acc = correct / max(1, total)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    support = cm.sum(axis=1)
    tp = np.diag(cm).astype(float)

    per_class_acc = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    valid_mask = (support > 0)
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
def build_text_features_simple(emotions, model, device, dtype=torch.float32):
    """
    Returns a [num_classes, d] matrix of L2-normalized CLIP text features
    for the given class labels (one prompt per class).
    """
    # model should already be on device; eval() is harmless if called again
    model.eval()

    # tokenize -> encode
    tokens = clip.tokenize(emotions).to(device)
    text_features = model.encode_text(tokens).to(dtype=dtype)

    # L2 normalize with numerical safety
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return text_features  # shape: [num_classes, d], typically d=512

class EmotionAwareAdapter(nn.Module):
    def __init__(self, hidden_dim=512, adapter_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, adapter_dim),
                nn.GELU(),
                nn.Dropout(dropout),          # <-- add dropout
                nn.Linear(adapter_dim, hidden_dim),
            ]
        self.adapter = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.adapter(x)

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

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

def compute_class_counts(train_loader, num_classes: int) -> torch.Tensor:
    """Count labels from the *training* loader on CPU with safe dtypes."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y, _ in train_loader:
        y = y.to('cpu', non_blocking=True).long()
        counts += torch.bincount(y, minlength=num_classes)
    return counts

def make_class_weights(counts: torch.Tensor, cap: float = 10.0) -> torch.Tensor:
    """Inverse-frequency weights, normalized & capped for stability."""
    counts = counts.clamp(min=1)  # avoid div-by-zero if a class is absent
    w = counts.sum().float() / counts.float()        # inverse freq
    w = (w / w.mean()).clamp(max=cap)               # normalize & cap
    return w


def train():
    # -- Setup --
    config = load_config('config.yaml')
    device = setup_device()

    gamma = config.get("gamma", 0.4)
    k = config.get("topk", 24)

    print(f"Training for gamma {gamma} and topk {k} ")
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
    # model.logit_scale.requires_grad = False  # keep frozen
    # # Unfreeze only the last transformer block of the visual encoder
    # for p in model.visual.transformer.resblocks[-1].parameters():
    #     p.requires_grad = True

    adapter = EmotionAwareAdapter(hidden_dim=512, adapter_dim=256, num_layers=2, dropout=0.2)
    for param in adapter.parameters():
        param.requires_grad = True

    adapter.to(device)

    # Set fixed temperature τ = 0.07
    import math
    with torch.no_grad():
        model.logit_scale.fill_(math.log(1.0 / 0.07))  # ≈ 14.29
    model.logit_scale.requires_grad_(False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)

    print("logit_scale.exp =", model.logit_scale.exp().item())

    emotions = ["neutral", "happy", "sad", "surprised", "fearful", "disgusted", "angry"]
    # text_features = build_text_features_simple(emotions, model, device)
    text_features = build_text_features_mean(by_class_prompt(emotions), model, device, emotions)
    text_features = text_features.detach().to(device)  # move to GPU/CPU once

    num_classes = config["num_classes"]
    # 1) Count labels on CPU (full train set)
    counts = compute_class_counts(train_loader, num_classes)  # returns cpu long tensor
    print("Train class counts:", counts.tolist())

    # 2) Stable inverse-frequency weights (normalized & capped)
    class_weights = make_class_weights(counts, cap=10.0).to(device)
    print("CE class weights:", class_weights.detach().cpu().tolist())

    # 3) Weighted CE
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.01  # keep small when using weights
    )

    # criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # small smoothing helps on noisy AffectNet

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['num_epochs']
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.to(device)

    # 4- Initialize metrics
    metrics_logger = MetricsLogger()


    # THIS IS FOR DEBUG
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Trainable (adapter): {adapter_trainable:,} | Trainable (model): {model_trainable:,} | "
          f"Frozen (model): {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    #THIS IS FOR DEBUG

    # 5 - Training loop
    best_val_loss = float('inf')
    best_mca = -float('inf')
    patience = 5
    early_stopping_counter = 0
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")

        train_loss, train_accuracy = train_epoch_adapter_only(train_loader, model, optimizer, criterion, device, text_features, epoch, adapter, scheduler, gamma, k)
        val_loss, val_acc, val_f1, val_cm, val_pc, val_mca, support = evaluate_last_block(val_loader, model, device, criterion, config['num_classes'], text_features, epoch, adapter, gamma, k)

        print("Current LR:", scheduler.get_last_lr()[0])

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

        print(f'best_mca: {best_mca:.4f}')
        print(f'val_mca: {val_mca:.4f}')

        # Save best model (by val_mca)
        monitor = val_mca

        if monitor > best_mca:
            best_mca = monitor
            early_stopping_counter = 0
            ckpt = {
                'epoch': epoch + 1,
                'adapter_state_dict': adapter.state_dict(),  # <-- add this
                'model_state_dict': model.state_dict(),  # CLIP (frozen)
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                "val_macro_f1": float(val_f1),
                "val_mean_class_acc": float(val_mca),
                "per_class_acc": [float(x) for x in val_pc],
                'config': config,
                'best_mca': float(best_mca)
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
        'adapter_state_dict': adapter.state_dict(),   # <-- add this
        'val_loss': val_loss,
        'val_acc': float(val_acc),
        'val_macro_f1': float(val_f1),
        'val_mean_class_acc': float(val_mca),
        'per_class_acc': [float(x) for x in val_pc],
        'config': config,
    }, f"{config['checkpoint_dir']}/last.pth")
import matplotlib.pyplot as plt

from data_baseline_freeze import labels_map_full

def plot_per_class_accuracy(per_class_accuracy, class_labels=None, title="Per-class Accuracy over Epochs"):
    """
    Plot per-class accuracy across epochs.

    Parameters
    ----------
    per_class_accuracy : list[list[float]]
        A list of epochs, each containing a list of per-class accuracies,
        e.g. shape [num_epochs][num_classes].
    class_labels : list[str] or None
        Optional list of class names (length = num_classes). If None, classes are indexed.
    title : str
        Plot title.
    """
    if not per_class_accuracy:
        raise ValueError("per_class_accuracy is empty.")

    num_epochs = len(per_class_accuracy)
    num_classes = len(per_class_accuracy[0])
    for i, ep in enumerate(per_class_accuracy):
        if len(ep) != num_classes:
            raise ValueError(f"Epoch {i} has {len(ep)} classes, expected {num_classes}.")

    if class_labels is None:
        class_labels = [f"Class {i+1}" for i in range(num_classes)]
    elif len(class_labels) != num_classes:
        raise ValueError(f"class_labels length {len(class_labels)} != num_classes {num_classes}.")

    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(8, 5))
    for class_idx in range(num_classes):
        series = [epoch_vals[class_idx] for epoch_vals in per_class_accuracy]
        plt.plot(epochs, series, marker='o', label=class_labels[class_idx])

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(epochs)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt

def plot_training_metrics(train_loss, mean_class_accuracy, save_path_prefix=None):
    """
    Plot training loss and mean class accuracy over epochs.

    Args:
        train_loss (list[float]): Training loss per epoch.
        mean_class_accuracy (list[float]): Mean class accuracy per epoch.
        save_path_prefix (str | None): If provided, saves figures as
            f"{save_path_prefix}_loss.png" and f"{save_path_prefix}_mca.png".
            (Saved to current working directory.)
    """
    epochs_loss = range(1, len(train_loss) + 1)
    epochs_mca  = range(1, len(mean_class_accuracy) + 1)

    # Plot 1: Training Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_loss, train_loss, marker='o', linestyle='-')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_loss.png", bbox_inches="tight", dpi=150)
    plt.show()

    # Plot 2: Mean Class Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_mca, mean_class_accuracy, marker='o', linestyle='-')
    plt.title("Mean Class Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Class Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_mca.png", bbox_inches="tight", dpi=150)
    plt.show()

def plot():
    metrics_logger = MetricsLogger()
    metrics_logger.load("./history-3/mfd-adjust/mfd_adjust-g-04-k-24/metrics.json")

    metrics_history = metrics_logger.get_metrics_history()
    plot_per_class_accuracy(metrics_history["per_class_accuracy"], labels_map_full)
    plot_training_metrics(metrics_history["train_loss"], metrics_history["mean_class_accuracy"])

if __name__ == '__main__':
    train()
    # plot()
