
from data_baseline_freeze import get_data_loaders_clip
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
def encode_image_tokens_vit(model, images, require_grad=True):
    """
    Returns:
      v_cls:     [B, D]        (projected CLS, normalized)
      v_patches: [B, N, D]     (projected patches, normalized)
    """
    visual = model.visual  # CLIP ViT
    x = visual.conv1(images)            # [B, C, H/patch, W/patch]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, N, width]
    x = torch.cat([visual.class_embedding.to(x.dtype)
                   + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    # Transformer (you may have un-frozen only the last block)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # ln_post then projection (apply to *all* tokens, not just CLS)
    x = visual.ln_post(x)                      # [B, 1+N, width]
    if hasattr(visual, "proj") and visual.proj is not None:
        x_proj = x @ visual.proj               # [B, 1+N, D]
    else:
        x_proj = x

    v_cls     = x_proj[:, 0, :]                # [B, D]
    v_patches = x_proj[:, 1:, :]               # [B, N, D]

    # Normalize to match text space
    v_cls     = v_cls / v_cls.norm(dim=-1, keepdim=True)
    v_patches = v_patches / v_patches.norm(dim=-1, keepdim=True)

    if not require_grad:
        v_cls = v_cls.detach()
        v_patches = v_patches.detach()

    return v_cls, v_patches

def train_epoch_last_block(train_loader, model, optimizer, criterion, device, text_features, epoch, mfd_head):
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
    top3 = []
    print("scale", scale)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # get CLS + patches (normalized)
        v_cls, v_patches = encode_image_tokens_vit(model, images, require_grad=True)  # [B,D], [B,N,D]

        # logits from MFD
        logits = mfd_head(v_cls, v_patches, text_features)  # [B, C]
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            print("Non-finite loss!", float(loss))
            print("NaN in v_cls:", torch.isnan(v_cls).any().item(), "v_patches:", torch.isnan(v_patches).any().item())
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        # metrics
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        batch_acc = (preds == labels).float().mean().item()
        ema_loss = update_ema(loss.item(), ema_loss)
        ema_acc  = update_ema(batch_acc, ema_acc)

        if not sample_logged:
            # Grad debug (handle None grads safely)
            last_block = model.visual.transformer.resblocks[-1]
            grad_norm_last = sum((p.grad.norm().item() for p in last_block.parameters() if p.grad is not None), 0.0)
            print(f"Grad Norm - Last Block: {grad_norm_last:.6f}")
            print("Label:     ", labels[:5].tolist())
            print("Prediction:", preds[:5].tolist())
            top3 = torch.topk(logits, k=3, dim=1)
            for i in range(min(5, images.size(0))):
                print(
                    f"Top-3 idx: {top3.indices[i].tolist()} | scores: {[round(x, 3) for x in top3.values[i].tolist()]}")
            sample_logged = True

        progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'ema_loss': f'{(ema_loss or 0):.3f}',
            'ema_acc': f'{(ema_acc or 0):.2f}',
            'lr': optimizer.param_groups[0]['lr'],
        })

    epoch_loss = running_loss / max(1, total)
    accuracy   = correct / max(1, total)
    return epoch_loss, accuracy

def mean_class_accuracy(per_class_acc):
    per_class_acc = np.asarray(per_class_acc, dtype=float)
    # ignore NaNs if any class has 0 support
    return np.nanmean(per_class_acc)

class MFDHead(torch.nn.Module):
    def __init__(self, k=16, gamma=0.1, tau=0.07):
        super().__init__()
        self.k = k
        self.gamma = gamma
        self.tau = tau

    def forward(self, v_cls, v_patches, text_features):
        # global affinities: [B, C]
        A_cls = torch.matmul(v_cls, text_features.T)

        # local affinities: [B, N, C]
        A_local = torch.matmul(v_patches, text_features.T)

        # top-k over patches -> mean: [B, C]
        A_local_topk = A_local.topk(self.k, dim=1).values.mean(dim=1)

        # combine global & local
        A_combined = self.gamma * A_cls + (1 - self.gamma) * A_local_topk

        logits = A_combined / self.tau
        return logits

@torch.no_grad()
def evaluate_last_block(val_loader, model, device, criterion, num_classes, text_features, mfd_head):
    """
    Evaluation for last-block fine-tuning setup.
    Uses fixed text_features for classification.
    """
    model.eval()
    mfd_head.eval()

    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # get CLS + patches (normalized)
            v_cls, v_patches = encode_image_tokens_vit(model, images, require_grad=True)  # [B,D], [B,N,D]

            # logits from MFD
            logits = mfd_head(v_cls, v_patches, text_features)  # [B, C]
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

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    mca = mean_class_accuracy(per_class_acc)

    return val_loss, val_acc, macro_f1, cm, per_class_acc, mca

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


    # 2.1 Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    model.logit_scale.requires_grad_(False)

    # 2.2 Unfreeze what you want to train
    last_block = model.visual.transformer.resblocks[-1]
    for p in last_block.parameters():
        p.requires_grad = True

    # Set fixed temperature τ = 0.07
    tau = 0.07
    with torch.no_grad():
        model.logit_scale.fill_(math.log(1.0 / tau))

    # 2.3 Optimizer param groups (encoder vs head, with sensible WD)
    enc_lr = config["learning_rate"]  # e.g., 5e-5 or 1e-4 | 1e-5
    head_lr = config.get("head_lr", 1e-3)  # usually higher for heads
    wd = config.get("weight_decay", 1e-2)
    param_groups = [
        {"params": last_block.parameters(), "lr": enc_lr, "weight_decay": wd},
    ]

    mfd_head = MFDHead(k=16, gamma=0.3, tau=tau).to(device)

    # add MFD head (don’t forget!)
    param_groups.append({"params": mfd_head.parameters(), "lr": head_lr, "weight_decay": wd})
    optimizer = torch.optim.AdamW(param_groups)

    # ONLY USED FOR EVAL
    emotions = ["neutral", "happy", "sad", "surprised", "fearful", "disgusted", "angry", "contemptuous"]
    text_features = build_text_features_simple(emotions, model, device)
    # text_features = build_text_features_mean(by_class_prompt(emotions), model, device, emotions).detach()

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

        train_loss, train_accuracy = train_epoch_last_block(train_loader, model, optimizer, criterion, device, text_features, epoch, mfd_head)
        val_loss, val_acc, val_f1, val_cm, val_pc, val_mca = evaluate_last_block(val_loader, model, device, criterion, config['num_classes'], text_features, mfd_head)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}:")
        print(f"Training   - Loss: {train_loss:.4f} | Acc: {train_accuracy:.2%}")

        print(f"Validation - Acc: {val_acc:.2%} | Macro-F1: {val_f1:.3f} | Mean Class Acc: {val_mca:.2%}")
        print("Per-class acc:", " ".join(f"{i}:{a:.2%}" for i, a in enumerate(val_pc)))

        # Log metrics
        metrics_logger.log_epoch(
            train_loss=train_loss,
            val_loss=val_loss,
            accuracy=train_accuracy,
            val_accuracy=val_acc,
            macro_f1=val_f1,
            mean_class_accuracy=val_mca
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

if __name__ == '__main__':
    train()
    # plot()
