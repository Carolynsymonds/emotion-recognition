
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_affectnet import get_data_loaders_clip
from tqdm import tqdm
from emotion_recognition.metrics import MetricsLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import clip
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from emotion_recognition.utils import load_config, setup_device, plot_metrics
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm

class EvidenceExtractor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden: int = 256,
                 max_evidence: float = 50.0):
        super().__init__()
        self.max_evidence = max_evidence

        if hidden is None:
            # Simple head: logits -> evidence
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, num_classes),
                nn.Softplus(beta=1.0, threshold=20.0),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, num_classes),
                nn.Softplus(beta=1.0, threshold=20.0),
            )

        self.reset_parameters()

    def reset_parameters(self):
        # keep initial evidence near zero (uncertain)
        last_linear = None
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.0)
                last_linear = m
        if last_linear is not None:
            # negative bias -> softplus(bias) ~ 0
            with torch.no_grad():
                last_linear.bias.add_(-4.0)  # softplus(-4) ≈ 0.018

    def forward(self, x):
        e = self.mlp(x)                    # non-negative
        if self.max_evidence is not None:  # cap to avoid huge Dirichlet alphas
            e = torch.clamp(e, 0.0, self.max_evidence)
        # tiny epsilon so alpha>1 strictly and log(probs) won’t hit -inf
        return e + 1e-8
def update_ema(val, ema, alpha=0.1):
    return val if ema is None else (alpha * val + (1 - alpha) * ema)

def kl_dirichlet(alpha):
    # KL( Dir(alpha) || Dir(1) ) ; stable, mean over batch
    S   = alpha.sum(1, keepdim=True)
    K   = alpha.size(1)
    logB = torch.lgamma(alpha).sum(1, keepdim=True) - torch.lgamma(S)
    logB0 = - torch.lgamma(torch.tensor([K], device=alpha.device))  # log B(1,...,1)
    dig   = torch.digamma(alpha) - torch.digamma(S)
    kl = (logB - logB0).squeeze(1) + ((alpha - 1.0) * dig).sum(1)
    return kl.mean()
def ruc_loss(p_list, u_list, labels, num_classes, eps=1e-8, reduce='mean'):
    """
    Relation Uncertainty Calibration loss (Eq. 13).
    p_list: list of [B, C] tensors (Dirichlet means per relation).
    u_list: list of [B] or [B,1] tensors (uncertainty mass per relation).
    """
    assert len(p_list) == len(u_list) and len(p_list) > 0
    device = labels.device
    B, C = labels.size(0), num_classes

    correct_accum = torch.tensor(0.0, device=device)
    wrong_accum   = torch.tensor(0.0, device=device)
    R = float(len(p_list))  # number of relations

    for p, u in zip(p_list, u_list):
        if u.dim() == 2:
            u = u.squeeze(1)  # [B]

        # predicted class & its probability
        yhat   = p.argmax(dim=1)                           # [B]
        p_pred = p.gather(1, yhat.unsqueeze(1)).squeeze(1) # [B]

        # masks
        correct = (yhat == labels)
        wrong   = ~correct

        # clamps
        u = u.clamp(min=eps, max=1.0 - eps)
        p_pred = p_pred.clamp(min=eps, max=1.0 - eps)

        # per-relation normalized terms (avoid skew when few samples qualify)
        if correct.any():
            term_c = (p_pred[correct] * torch.log(1.0 - u[correct])).mean()
            correct_accum = correct_accum + term_c
        if wrong.any():
            term_w = ((1.0 - p_pred[wrong]) * torch.log(u[wrong])).mean()
            wrong_accum = wrong_accum + term_w

    # average over relations; apply 1/C factor to the wrong term (per Eq. 13)
    loss = -( correct_accum / R + (wrong_accum / R) / C )

    if reduce == 'sum':
        loss = loss * B  # optional
    return loss
def project_visual_tokens(clip_model, tokens_768):
    x = clip_model.visual.ln_post(tokens_768)
    if clip_model.visual.proj is not None:
        x = x @ clip_model.visual.proj
    return x
def train_epoch_last_block(train_loader, model, class_weights, optimizer, device, text_features, epoch, adapter, scheduler, evidence_extractor_g, num_classes):
    """
     Train the Emotion-aware Adapter on top of frozen CLIP visual encoder.
    Use fixed text_features for classification.
    """
    model.eval()  # Freeze CLIP backbone
    adapter.train()  # Train adapter
    evidence_extractor_g.train()

    # hygiene
    assert text_features.size(0) == num_classes
    class_weights = class_weights.to(device).float()

    running_loss, correct, n_seen = 0.0, 0, 0
    ema_loss, ema_acc = None, None
    with torch.no_grad():
        text_features = F.normalize(text_features, dim=1)
    text_features.requires_grad_(False)

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    sample_logged = False
    lambda_kl_max = 0.001
    lambda_ruc = 0.05
    # Anneal lambda_kl (warm-up over 10 epochs)
    lambda_kl = min(lambda_kl_max * (epoch / 10), lambda_kl_max)

    print("Logit scale (TO TEST):", model.logit_scale.exp().item())
    epoch_pred_counts = torch.zeros(num_classes, dtype=torch.long)  # <-- before loop

    for step, (images, labels) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ----- Frozen CLIP feature extraction -----
        with torch.no_grad():
            patch_tokens_768 = extract_patch_tokens(model, images)  # [B, N+1, 768]
            projected_tokens = project_visual_tokens(model, patch_tokens_768)  # [B, N+1, 512]

        # ----- Adapter forward -----
        tokens = adapter(projected_tokens)  # [B, N+1, 512]
        v_cls = tokens[:, 0, :]  # [B, 512]

        # ----- Global logits -----
        scale = model.logit_scale.exp()
        logits_g = scale * (F.normalize(v_cls, dim=1) @ text_features.t())  # [B, C]

        # ----- Evidential (global) -----
        e_g = F.softplus(evidence_extractor_g(logits_g))  # ensure e >= 0
        alpha_g = e_g + 1.0
        Sg = alpha_g.sum(1, keepdim=True)
        p_g = (alpha_g / Sg).clamp_min(1e-8)
        u_g = (num_classes / Sg).clamp(1e-6, 1-1e-6)

        # ----- Losses -----
        edl_ce = F.nll_loss(torch.log(p_g + 1e-8), labels, weight=class_weights)
        edl_kl = kl_dirichlet(alpha_g)
        L_EDL = edl_ce + lambda_kl * edl_kl

        # RUC (global-only)
        L_RUC = ruc_loss([p_g], [u_g], labels, num_classes=num_classes)
        loss = L_EDL + lambda_ruc * L_RUC

        # ----- Backward -----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) +
                                       list(evidence_extractor_g.parameters()), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # ----- Metrics -----
        running_loss += loss.item() * images.size(0)
        preds = p_g.argmax(dim=1)  # stay on device
        correct += (preds == labels).sum().item()
        n_seen += labels.size(0)

        # histogram (CPU)
        epoch_pred_counts += torch.bincount(preds.cpu(), minlength=num_classes)

        batch_acc = (preds == labels).float().mean().item()
        ema_loss = update_ema(loss.item(), ema_loss)
        ema_acc = update_ema(batch_acc, ema_acc)

        if step % 100 == 0:
            print("Predicted class:", preds[0].item())
            print("Class probabilities:", p_g[0].detach().cpu().numpy())
            print("Uncertainty:", float(u_g[0].detach().cpu()))

            total_so_far = int(epoch_pred_counts.sum().item())
            if total_so_far > 0:
                perc = (epoch_pred_counts.float() / total_so_far * 100.0).tolist()
                print("Pred counts per class (so far):", epoch_pred_counts.tolist())
                print("Pred % (so far):", [round(p, 1) for p in perc])

        if not sample_logged:
            print("Sample labels:", labels[:5].tolist())
            print("Sample predictions:", preds[:5].tolist())
            top3 = torch.topk(p_g, k=3, dim=1)
            for i in range(min(5, images.size(0))):
                print(f"Label: {labels[i].item()}, Top-3: {top3.indices[i].tolist()}, "
                      f"Scores: {top3.values[i].tolist()}")
            sample_logged = True

        progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'ema_loss': f'{ema_loss:.3f}',
            'ema_acc': f'{ema_acc:.2f}',
            'lr': optimizer.param_groups[0]['lr']
        })

    # ----- DDP reduce histogram (and optionally accuracy) -----
    if torch.distributed.is_initialized():
        counts = epoch_pred_counts.to(device)
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        epoch_pred_counts = counts.cpu()

        # optional: global accuracy (uncomment if needed)
        # buf = torch.tensor([correct, n_seen], device=device, dtype=torch.long)
        # torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
        # correct, n_seen = int(buf[0].item()), int(buf[1].item())

    total_preds = int(epoch_pred_counts.sum().item())
    perc_epoch = (epoch_pred_counts.float() / max(1, total_preds) * 100.0).tolist()
    print("Pred counts per class (epoch):", epoch_pred_counts.tolist())
    print("Pred % (epoch):", [round(p, 1) for p in perc_epoch])

    epoch_loss = running_loss / max(1, n_seen)
    accuracy = correct / max(1, n_seen)
    return epoch_loss, accuracy

def mean_class_accuracy(per_class_acc):
    per_class_acc = np.asarray(per_class_acc, dtype=float)
    # ignore NaNs if any class has 0 support
    return np.nanmean(per_class_acc)

@torch.no_grad()
def evaluate_last_block_global(
    val_loader, model, device, num_classes, text_features,
    adapter, evidence_extractor, class_weights=None,
    lambda_kl: float = 0.0, lambda_ruc: float = 0.0
):
    model.eval(); adapter.eval(); evidence_extractor.eval()

    assert text_features.size(0) == num_classes
    text_features = F.normalize(text_features.to(device), dim=1)

    scale = model.logit_scale.exp()
    C = num_classes

    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    epoch_pred_counts = torch.zeros(num_classes, dtype=torch.long)  # CPU histogram

    # (optional) class weights for loss comparability with train
    if class_weights is not None:
        class_weights = class_weights.to(device).float()

    for images, labels in val_loader:
        images = images.to(device).float()
        labels = labels.to(device)

        # ----- Frozen CLIP features -----
        t768 = extract_patch_tokens(model, images)       # [B, N+1, 768]
        t512 = project_visual_tokens(model, t768)        # [B, N+1, 512]

        # ----- Adapter -----
        tokens = adapter(t512)                           # [B, N+1, 512]
        v_cls  = tokens[:, 0, :]

        # ----- Global logits -----
        v = F.normalize(v_cls, dim=1)
        logits_g = scale * (v @ text_features.t())      # [B, C]

        # ----- Evidence -> Dirichlet -----
        e_g     = evidence_extractor(logits_g)          # [B, C]
        alpha_g = e_g + 1.0
        Sg      = alpha_g.sum(1, keepdim=True)
        p_g     = alpha_g / Sg                           # [B, C]
        u_g     = C / Sg                                 # [B, 1] (unused unless lambda_ruc>0)

        # ----- Loss (match train settings if desired) -----
        ce = F.nll_loss(torch.log(p_g + 1e-8), labels, weight=class_weights) \
             if class_weights is not None else \
             F.nll_loss(torch.log(p_g + 1e-8), labels)

        loss = ce
        if lambda_kl > 0:
            loss = loss + lambda_kl * kl_dirichlet(alpha_g)
        if lambda_ruc > 0:
            loss = loss + lambda_ruc * ruc_loss([p_g], [u_g], labels, num_classes=C)

        # ----- Metrics -----
        val_loss += loss.item() * images.size(0)
        preds = p_g.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # accumulate for later sklearn metrics
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        # accumulate histogram (CPU)
        epoch_pred_counts += torch.bincount(preds.cpu(), minlength=num_classes)

    # ---- DDP reductions (optional but recommended) ----
    if torch.distributed.is_initialized():
        counts = epoch_pred_counts.to(device)
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        epoch_pred_counts = counts.cpu()

        buf = torch.tensor([correct, total, val_loss], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
        correct = int(buf[0].item()); total = int(buf[1].item()); val_loss = float(buf[2].item())

    # histogram summary
    total_preds = int(epoch_pred_counts.sum().item())
    perc_epoch = (epoch_pred_counts.float() / max(1, total_preds) * 100.0).tolist()
    print("Pred counts per class (validation epoch):", epoch_pred_counts.tolist())
    print("Pred % (validation epoch):", [round(p, 1) for p in perc_epoch])

    # sklearn-style metrics
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_loss /= max(1, total)
    val_acc   = correct / max(1, total)

    cm       = confusion_matrix(all_labels, all_preds, labels=np.arange(C))
    support  = cm.sum(axis=1)
    tp       = np.diag(cm).astype(float)
    per_class_acc = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    valid_mask    = (support > 0)
    macro_f1      = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mca           = float(per_class_acc[valid_mask].mean()) if valid_mask.any() else 0.0

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

def compute_class_counts(train_loader, num_classes: int) -> torch.Tensor:
    """Count labels from the *training* loader on CPU with safe dtypes."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y in train_loader:
        y = y.to('cpu', non_blocking=True).long()
        counts += torch.bincount(y, minlength=num_classes)
    return counts

def make_class_weights(
    counts: torch.Tensor,
    cap: float = 10.0,
    scheme: str = "inv",          # "inv" or "cb"
    alpha: float = 1.0,           # soften 1/f by raising to alpha (0.5–1.0 typical)
    cb_beta: float = 0.999        # for "cb" scheme
) -> torch.Tensor:
    """
    Returns class weights normalized to mean=1 and capped to [1/cap, cap].
    counts: per-class counts from the FULL TRAIN SET (on CPU is fine).
    """
    counts = counts.clamp(min=1).float()

    if scheme == "inv":
        w = (counts.sum() / counts).pow(alpha)
    elif scheme == "cb":
        # Class-Balanced: w_c ∝ (1 - β) / (1 - β^{n_c})
        eff_num = 1.0 - cb_beta**counts
        w = (1.0 - cb_beta) / eff_num
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    w = w / w.mean()                     # normalize scale
    w = w.clamp(min=1.0/cap, max=cap)    # cap extremes
    return w



import torch
import torch.nn.functional as F

def dump_epoch_predictions_global(
    loader,
    model,
    adapter,
    evidence_extractor_g,
    text_features,
    device,
    class_names=None,
    save_csv_path=None,
    max_samples: int = 10,
    verbose: bool = True,
):
    """
    Collect per-image predictions & uncertainties (global-only RUC) for up to `max_samples` images.
    Prints which images were logged. Returns the collected rows list.
    """
    model.eval(); adapter.eval(); evidence_extractor_g.eval()
    text_features = F.normalize(text_features.to(device), dim=1)
    C = text_features.size(0)
    scale = model.logit_scale.exp()

    rows = []
    idx_counter = 0
    done = False

    with torch.no_grad():
        for batch in loader:
            # Accept (img, label) or (img, label, path)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = [None] * images.size(0)

            images = images.to(device).float()
            labels = labels.to(device)

            # ----- frozen CLIP -----
            tok_768 = extract_patch_tokens(model, images)      # [B, N+1, 768]
            tok_512 = project_visual_tokens(model, tok_768)    # [B, N+1, 512]

            # ----- adapter -----
            tokens = adapter(tok_512)                          # [B, N+1, 512]
            v_cls  = tokens[:, 0, :]                           # [B, 512]

            # ----- logits_g (CLS ↔ text) -----
            v = F.normalize(v_cls, dim=1)
            logits_g = scale * (v @ text_features.t())         # [B, C]

            # ----- Dirichlet (global) -----
            e_g     = evidence_extractor_g(logits_g)           # [B, C], Softplus in module
            alpha_g = e_g + 1.0                                 # [B, C]
            Sg      = alpha_g.sum(1, keepdim=True)              # [B, 1]
            p_g     = alpha_g / Sg                              # [B, C]
            u_g     = (C / Sg).squeeze(1)                       # [B]

            # how many from this batch to log
            B = images.size(0)
            remaining = max_samples - len(rows)
            take = min(remaining, B)

            for b in range(take):
                pred = int(torch.argmax(p_g[b]).item())
                gt   = int(labels[b].item())
                topk = torch.topk(p_g[b], k=min(3, C))
                row = {
                    "idx": idx_counter,
                    "path": paths[b] if isinstance(paths[b], str) else None,
                    "gt": gt,
                    "gt_name": (class_names[gt] if class_names else str(gt)),
                    "pred": pred,
                    "pred_name": (class_names[pred] if class_names else str(pred)),
                    "correct": bool(pred == gt),
                    "p_pred": float(p_g[b, pred].item()),
                    "u": float(u_g[b].item()),
                    "alpha_sum": float(alpha_g[b].sum().item()),
                    "alpha_pred": float(alpha_g[b, pred].item()),
                    "alpha": [float(x) for x in alpha_g[b].tolist()],
                    "probs": [float(x) for x in p_g[b].tolist()],
                    "top3_idx": [int(i) for i in topk.indices.tolist()],
                    "top3_prob": [float(v) for v in topk.values.tolist()],
                }
                rows.append(row)
                idx_counter += 1

            if len(rows) >= max_samples:
                done = True
                break

        # stop outer loop if we hit the cap
        if not done:
            pass

    # Optional: write CSV
    if save_csv_path is not None:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(save_csv_path, index=False)
            print(f"[dump_epoch_predictions_global] wrote {len(rows)} rows to {save_csv_path}")
        except Exception as e:
            print(f"[dump_epoch_predictions_global] CSV save skipped ({e})")

    # Verbose summary of exactly which images were logged
    if verbose:
        print("\n=== Logged samples (max 10) ===")
        for r in rows:
            ident = r["path"] if r["path"] is not None else f"idx#{r['idx']}"
            print(f"{ident} | GT={r['gt_name']}  PRED={r['pred_name']}  "
                  f"p={r['p_pred']:.3f}  u={r['u']:.3f}  correct={r['correct']}")

    return rows


@torch.no_grad()
def probe_label_prompt_alignment(
    loader, model, adapter, text_features, device, class_names, max_batches=8
):
    model.eval(); adapter.eval()
    tf = F.normalize(text_features.to(device), dim=1)
    C  = tf.size(0)
    assert len(class_names) == C, "class_names length must match text_features rows"

    scale = model.logit_scale.exp()

    # Accumulate on CPU
    M   = torch.zeros(C, C, dtype=torch.float64)   # [true_label, prompt_idx]
    cnt = torch.zeros(C, dtype=torch.long)

    for bi, batch in enumerate(loader):
        images, labels = batch[:2]
        images = images.to(device).float()
        labels = labels.to(device)

        # frozen CLIP -> adapter -> CLS
        tok_768 = extract_patch_tokens(model, images)
        tok_512 = project_visual_tokens(model, tok_768)
        v_cls   = adapter(tok_512)[:, 0, :]
        v       = F.normalize(v_cls, dim=1)

        # global logits -> softmax over prompts
        probs = (scale * (v @ tf.t())).softmax(dim=1)  # [B, C], on GPU

        # accumulate per true label (move addend to CPU)
        for i in range(C):
            m = (labels == i)
            if m.any():
                M[i] += probs[m].sum(dim=0).double().cpu()
                cnt[i] += int(m.sum().item())

        if bi + 1 >= max_batches:
            break

    # row-normalize by counts
    for i in range(C):
        if cnt[i] > 0:
            M[i] /= cnt[i].item()

    # report top-3 per true label
    print("\n[Probe] avg prob per (true_label → prompt_idx)")
    for i in range(C):
        row = M[i]
        topv, topi = torch.topk(row, k=min(3, C))
        tops = ", ".join([f"{int(topi[k])}({class_names[int(topi[k])]})={float(topv[k]):.3f}"
                          for k in range(len(topi))])
        print(f"true {i} ({class_names[i]}): {tops}")

    suggested = M.argmax(dim=1).tolist()
    print("\nSuggested mapping (label i → prompt idx):", suggested)
    if suggested == list(range(C)):
        print("✅ Mapping looks correct (identity).")
    elif sorted(suggested) == list(range(C)):
        print("⚠️ Mapping is a permutation. You can fix by: text_features = text_features[suggested]")
    else:
        print("⚠️ Not a permutation—prompts likely overlap. Expand/improve templates.")

    return M, cnt, suggested

def train():
    # -- Setup --
    config = load_config('config.yaml')
    device = setup_device()

    gamma = config.get("gamma", 0.1)
    k = config.get("topk", 32)

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


    print("logit_scale.exp =", model.logit_scale.exp().item())

    emotions = ["neutral", "happy", "sad", "surprised", "fearful", "disgusted", "angry"]
    # text_features = build_text_features_simple(emotions, model, device)
    text_features = build_text_features_mean(by_class_prompt(emotions), model, device, emotions)
    text_features = text_features.detach().to(device)  # move to GPU/CPU once

    num_classes = config["num_classes"]
    # 1) Count labels on CPU (full train set)
    counts = compute_class_counts(train_loader, num_classes)  # returns cpu long tensor
    print("Train class counts:", counts.tolist())

    print("\n[Label ↔ Prompt mapping in use]")
    for i, name in enumerate(emotions):
        print(f"label {i}: {name}")

    _ = probe_label_prompt_alignment(val_loader, model, adapter, text_features, device, emotions, max_batches=8)

    # 2) Stable inverse-frequency weights (normalized & capped)
    class_weights = make_class_weights(counts, cap=10.0, scheme="inv", alpha=0.8).to(device).float()
    class_weights = class_weights / class_weights.mean()  # stabilizes loss scale

    print("CE class weights:", class_weights.detach().cpu().tolist())

    # 3) Weighted CE
    # criterion = nn.CrossEntropyLoss(
    #     weight=class_weights,
    #     label_smoothing=0.01  # keep small when using weights
    # )

    # YOUR model’s label order (example from earlier)


    # criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # small smoothing helps on noisy AffectNet


    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['num_epochs']
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
        # optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)

    evidence_extractor = EvidenceExtractor(num_classes, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(evidence_extractor.parameters()),
        lr=config["learning_rate"], weight_decay=1e-4
    )

    torch.nn.utils.clip_grad_norm_(
        list(adapter.parameters()) + list(evidence_extractor.parameters()),
        1.0
    )
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
    best_mca = -float('inf')
    patience = 5
    early_stopping_counter = 0
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")

        train_loss, train_accuracy = train_epoch_last_block(train_loader, model, class_weights, optimizer, device, text_features, epoch, adapter, scheduler, evidence_extractor, config['num_classes'])
        val_loss, val_acc, val_f1, val_cm, val_pc, val_mca, support = evaluate_last_block_global(val_loader, model, device, config['num_classes'], text_features, adapter, evidence_extractor)


        # after training + validation for the epoch:
        rows = dump_epoch_predictions_global(
            val_loader,
            model,
            adapter,
            evidence_extractor,
            text_features,
            device,
            class_names=["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"],
            save_csv_path=f"epoch_{epoch:03d}_val_preds_ruc.csv",  # or None
        )

        # quick sanity: show 5 mistakes with highest confidence (low u, high p_pred)
        mistakes = [r for r in rows if not r["correct"]]
        mistakes = sorted(mistakes, key=lambda r: (-r["p_pred"], r["u"]))[:5]
        for r in mistakes:
            print(f"[MIS] idx={r['idx']} path={r['path']} GT={r['gt_name']} PRED={r['pred_name']} "
                  f"p={r['p_pred']:.3f} u={r['u']:.3f}")

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
        metrics_logger.save('metrics_parallel.json')

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
    import math
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
    target_names = ["7.jpg", "16.jpg", "17.jpg", "36.jpg", "3.jpg", "19.jpg", "1008.jpg"]
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

def att_heatmap():
    global device, p
    config = load_config('config.yaml')
    device = setup_device()
    train_loader, val_loader, test_loader = get_data_loaders_clip(config, device)
    import torch
    ckpt_path = "./history-3/mfd-adjust/best_model.pth"  # or config['checkpoint_dir']/best_model.pth
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
    import math
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