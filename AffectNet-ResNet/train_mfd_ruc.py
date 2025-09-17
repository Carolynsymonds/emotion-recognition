
from data_baseline_freeze import get_data_loaders_clip
from tqdm import tqdm
from metrics import MetricsLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import clip
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from utils import load_config, setup_device, plot_metrics
import torch.nn as nn
import torch
import torch.nn.functional as F

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
def kl_dirichlet(alpha):
    # KL( Dir(alpha) || Dir(1) ) ; stable, mean over batch
    S   = alpha.sum(1, keepdim=True)
    K   = alpha.size(1)
    logB = torch.lgamma(alpha).sum(1, keepdim=True) - torch.lgamma(S)
    logB0 = - torch.lgamma(torch.tensor([K], device=alpha.device))  # log B(1,...,1)
    dig   = torch.digamma(alpha) - torch.digamma(S)
    kl = (logB - logB0).squeeze(1) + ((alpha - 1.0) * dig).sum(1)
    return kl.mean()
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
def ruc_loss(p_list, u_list, labels, num_classes, eps=1e-8):
    """
    Implements Eq. (13):
    L_RUC = - 1/N * sum_i [ sum_{j in correct}   p_i^(j) * log(1 - u_i^(j))
                           + (1/C) * sum_{j in wrong} (1 - p_i^(j)) * log(u_i^(j)) ]

    Args:
      p_list: list of tensors, each [B, C], expected class probs for each relation (e.g., [p_g, p_l]).
      u_list: list of tensors, each [B] or [B,1], uncertainty scalars for each relation (u = C / sum(alpha)).
      labels: LongTensor [B].
      num_classes: int, C.
    """
    assert len(p_list) == len(u_list)
    B = labels.size(0)
    C = num_classes

    total_correct_term = 0.0
    total_wrong_term   = 0.0

    for p, u in zip(p_list, u_list):
        if u.dim() == 2:
            u = u.squeeze(1)                      # [B]

        # relation's predicted label and its probability
        yhat = p.argmax(dim=1)                    # [B]
        p_pred = p.gather(1, yhat.unsqueeze(1)).squeeze(1)  # [B]

        # masks: which samples this relation predicts correctly / wrongly
        correct = (yhat == labels)                # [B] bool
        wrong   = ~correct

        # clamp for numerical safety
        u_clamp = u.clamp(min=eps, max=1 - eps)

        # terms from Eq. (13)
        if correct.any():
            total_correct_term += (p_pred[correct] * torch.log(1.0 - u_clamp[correct])).sum()
        if wrong.any():
            total_wrong_term   += ((1.0 - p_pred[wrong]) * torch.log(u_clamp[wrong])).sum()

    # average over batch; wrong term normalized by C (the equation’s 1/C factor)
    L = -( total_correct_term + (total_wrong_term / C) ) / max(1, B)
    return L

def KL(alpha):
    K = alpha.size(1)
    beta = torch.ones((1, K), device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    KL_div = (
        torch.lgamma(S_alpha).sum()
        - torch.lgamma(alpha).sum(dim=1).sum()
        + torch.lgamma(beta).sum()
        - torch.lgamma(torch.sum(beta, dim=1))
        + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1)
    )
    return KL_div.mean()
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

    running_loss, correct, total = 0.0, 0, 0
    ema_loss, ema_acc = None, None

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
        C = text_features.size(0)

        # ----- Evidence (global) -----
        e_g = evidence_extractor_g(logits_g)  # [B, C], Softplus inside
        alpha_g = e_g + 1.0  # Dirichlet params
        Sg = alpha_g.sum(1, keepdim=True)  # [B, 1]
        p_g = alpha_g / Sg  # expected probs [B, C]
        u_g = C / Sg  # uncertainty [B, 1]

        # ----- Losses -----
        edl_ce = F.nll_loss(torch.log(p_g + 1e-8), labels, weight=class_weights)
        edl_kl = kl_dirichlet(alpha_g)
        L_EDL = edl_ce + lambda_kl * edl_kl

        # RUC (Eq. 13) with only the global relation
        L_RUC = ruc_loss([p_g], [u_g], labels, num_classes=C)

        loss = L_EDL + lambda_ruc * L_RUC

        # ----- Backward & optimize -----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(adapter.parameters()) + list(evidence_extractor_g.parameters()),
            1.0
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # ----- Metrics (use calibrated probs) -----
        running_loss += loss.item() * images.size(0)
        preds = p_g.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        batch_acc = (preds == labels).float().mean().item()
        ema_loss = update_ema(loss.item(), ema_loss)
        ema_acc = update_ema(batch_acc, ema_acc)


        if step % 100 == 0:
            print("Pred counts per class (so far):", epoch_pred_counts.tolist())
            perc = (epoch_pred_counts.float() / epoch_pred_counts.sum()).tolist()
            print("Pred % (so far):", [round(100 * p, 1) for p in perc])

        if not sample_logged:
            print("Sample labels:", labels[:5].tolist())
            print("Sample predictions:", preds[:5].tolist())
            top3 = torch.topk(p_g, k=3, dim=1)
            for i in range(min(5, images.size(0))):
                print(
                    f"Label: {labels[i].item()}, Top-3: {top3.indices[i].tolist()}, Scores: {top3.values[i].tolist()}")
            sample_logged = True

        progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'ema_loss': f'{ema_loss:.3f}',
            'ema_acc': f'{ema_acc:.2f}',
            'lr': optimizer.param_groups[0]['lr']
        })

    print("Pred counts per class (epoch):", epoch_pred_counts.tolist())

    epoch_loss = running_loss / max(1, total)
    accuracy   = correct / max(1, total)
    return epoch_loss, accuracy

def mean_class_accuracy(per_class_acc):
    per_class_acc = np.asarray(per_class_acc, dtype=float)
    # ignore NaNs if any class has 0 support
    return np.nanmean(per_class_acc)

@torch.no_grad()
def evaluate_last_block_global(
    val_loader, model, device, num_classes, text_features,
    adapter, evidence_extractor
):
    """
    Evaluation for last-block setup with global-only RUC.
    Uses CLIP CLS->text global logits, then evidence->Dirichlet.
    Returns metrics using calibrated probs (expected Dirichlet).
    """
    model.eval()
    adapter.eval()
    evidence_extractor.eval()

    lambda_kl = 0.0
    lambda_ruc = 0.0

    assert text_features.size(0) == num_classes, \
        f"num_classes ({num_classes}) must match text_features.size(0) ({text_features.size(0)})"
    text_features = F.normalize(text_features.to(device), dim=1)  # ensure L2-norm

    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    C = num_classes
    scale = model.logit_scale.exp()  # scalar tensor
    epoch_pred_counts = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device).float()
            labels = labels.to(device)

            # ----- Frozen CLIP features -----
            tokens_768 = extract_patch_tokens(model, images)         # [B, N+1, 768]
            tokens_512 = project_visual_tokens(model, tokens_768)    # [B, N+1, 512]

            # ----- Adapter -----
            tokens = adapter(tokens_512)                             # [B, N+1, 512]
            v_cls  = tokens[:, 0, :]                                 # [B, 512]

            # ----- Global logits (CLS ↔ text) -----
            v_cls_n   = F.normalize(v_cls, dim=1)
            logits_g  = scale * (v_cls_n @ text_features.t())        # [B, C]

            # ----- Evidence -> Dirichlet -----
            e_g     = evidence_extractor(logits_g)                   # [B, C], Softplus inside
            alpha_g = e_g + 1.0                                      # [B, C]
            Sg      = alpha_g.sum(1, keepdim=True)                   # [B, 1]
            p_g     = alpha_g / Sg                                   # expected probs [B, C]
            u_g     = C / Sg                                         # uncertainty [B, 1]

            # ----- Losses -----
            ce = F.nll_loss(torch.log(p_g + 1e-8), labels)
            loss = ce
            if lambda_kl > 0:
                loss = loss + lambda_kl * kl_dirichlet(alpha_g)      # or KL(alpha_g) if that's your name
            if lambda_ruc > 0:
                loss = loss + lambda_ruc * ruc_loss([p_g], [u_g], labels, num_classes=C)

            # ----- Accumulate -----
            val_loss += loss.item() * images.size(0)
            preds = p_g.argmax(dim=1)                                # calibrated preds
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    print("Pred counts per class (validation epoch):", epoch_pred_counts.tolist())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_loss /= max(1, total)
    val_acc   = correct / max(1, total)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm       = confusion_matrix(all_labels, all_preds, labels=np.arange(C))
    support  = cm.sum(axis=1)
    tp       = np.diag(cm).astype(float)
    per_class_acc = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    valid_mask    = (support > 0)
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
def probe_label_prompt_alignment(loader, model, adapter, text_features, device, class_names, max_batches=8):
    model.eval(); adapter.eval()
    tf = F.normalize(text_features.to(device), dim=1)
    C  = tf.size(0)
    scale = model.logit_scale.exp()
    M = torch.zeros(C, C, dtype=torch.float64)
    cnt = torch.zeros(C, dtype=torch.long)

    for bi, batch in enumerate(loader):
        images, labels = batch[:2]
        images = images.to(device).float()
        labels = labels.to(device)

        tok_768 = extract_patch_tokens(model, images)
        tok_512 = project_visual_tokens(model, tok_768)
        v_cls   = adapter(tok_512)[:, 0, :]
        v       = F.normalize(v_cls, dim=1)

        probs = (scale * (v @ tf.t())).softmax(dim=1)  # [B, C]
        for i in range(C):
            m = (labels == i)
            if m.any():
                M[i] += probs[m].sum(dim=0).double()
                cnt[i] += int(m.sum())

        if bi + 1 >= max_batches:
            break

    for i in range(C):
        if cnt[i] > 0: M[i] /= cnt[i].item()

    print("\n[Probe] avg prob per (true_label → prompt_index)")
    for i in range(C):
        row = M[i]
        topv, topi = torch.topk(row, k=min(3, C))
        tops = ", ".join([f"{int(topi[k])}({class_names[int(topi[k])]})={topv[k]:.3f}" for k in range(len(topi))])
        print(f"true {i} ({class_names[i]}): {tops}")

    suggested = M.argmax(dim=1).tolist()
    print("\nSuggested mapping (label i → prompt idx):", suggested)
    if suggested == list(range(C)):
        print("✅ Mapping looks correct (identity).")
    else:
        # If it's a permutation, you can reindex:
        if sorted(suggested) == list(range(C)):
            print("⚠️ Reindexing text_features by suggested permutation.")
            # text_features = text_features[suggested]   # do this if you want to auto-fix
        else:
            print("⚠️ Not a permutation. Prompts may be too overlapping; expand templates.")
    return M, cnt, suggested

def train():
    # -- Setup --
    config = load_config('config.yaml')
    device = setup_device()

    gamma = config.get("gamma", 0.4)
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
        lr=1e-4, weight_decay=1e-4
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

if __name__ == '__main__':
    train()