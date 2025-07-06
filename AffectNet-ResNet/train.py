import timm
import torch
from data import get_data_loaders, get_data_loaders_clip
from tqdm import tqdm
from utils import load_config, setup_device, plot_metrics
import torch.optim as optim
from torchsummary import summary
from metrics import MetricsLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import clip


def train(model, train_loader, optimizer, device, epoch, loss_function):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    sample_logged = False

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Accuracy calculation
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # TEST Sample
        if not sample_logged:
            print("Label: {}".format(labels.tolist()))
            print("Prediction: {}".format(predicted.tolist()))
            sample_logged = True

        progress_bar.set_postfix(
            {
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            },
        )

    epoch_loss = running_loss / total
    accuracy = correct / total

    return epoch_loss, accuracy

def validate(model, val_loader, device, loss_function):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss / total
    accuracy = correct / total
    return val_loss, accuracy

def train_clip(model, train_loader, optimizer, device, epoch, loss_function, text_features):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    sample_logged = False

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T

        loss = loss_function(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # TEST Sample
        if not sample_logged:
            print("Label: {}".format(labels.tolist()))
            print("Prediction: {}".format(predicted.tolist()))
            sample_logged = True

        progress_bar.set_postfix(
            {
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            },
        )

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

def validate_clip(model, val_loader, device, loss_function, text_features):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T
            loss = loss_function(logits, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss / total
    accuracy = correct / total
    return val_loss, accuracy


if __name__ == '__main__':

    train_model = "clip"

    # 1 - Setup
    config = load_config('config.yaml')
    device = setup_device()

    print(f"Training for {config['num_epochs']} epochs")

    # 2. Define Model, Loss, and Optimizer
    if train_model == "clip":
        train_loader, val_loader, test_loader = get_data_loaders_clip(config, device)

        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.float()
        for param in model.parameters():
            param.requires_grad = False
        # for name, param in model.named_parameters():
        #     if "visual.proj" in name:
        #         param.requires_grad = True
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        emotion_classes = [
            "a photo of a neutral face",
            "a photo of a happy face",
            "a photo of a sad face",
            "a photo of a surprised face",
            "a photo of a fearful face",
            "a photo of a disgusted face",
            "a photo of an angry face",
            "a photo of a contemptuous face"
        ]
        text_tokens = clip.tokenize(emotion_classes).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    else:
        train_loader, val_loader, test_loader = get_data_loaders(config)
        model = timm.create_model(
            'efficientvit_m5.r224_in1k',
            pretrained=True,
            num_classes=8  # set this to your number of emotion classes
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7, weight_decay=1e-4)

    loss_function = torch.nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    model.to(device)

    # VIEW summary of the model
    # INPUT_SHAPE = (3, 256, 256)
    # print(summary(model, (INPUT_SHAPE)))

    # 4- Initialize metrics
    metrics_logger = MetricsLogger()

    # 5 - Training loop
    best_val_loss = float('inf')
    patience = 5
    early_stopping_counter = 0
    for epoch in range(config['num_epochs']):
        # Training
        if train_model == "clip":
            train_loss, train_accuracy = train_clip(
                model, train_loader, optimizer, device, epoch, loss_function, text_features)
            val_loss, val_accuracy = validate_clip(model, val_loader, device, loss_function, text_features)
        else:
            train_loss, train_accuracy = train(
                model, train_loader, optimizer, device, epoch, loss_function)
            # Validation
            val_loss, val_accuracy = validate(model, val_loader, device, loss_function)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}:")
        print(f"Training     - Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.2%}")
        print(f"Validation   - Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.2%}")

        # Log metrics
        metrics_logger.log_epoch(train_loss=train_loss, val_loss=val_loss, accuracy=train_accuracy, val_accuracy=val_accuracy)
        history = metrics_logger.get_metrics_history()
        metrics_logger.save('metrics.json')
        print(f'best val loss: {best_val_loss}')
        print(f'val loss: {val_loss}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), f"{config['checkpoint_dir']}/best_model.pth")
            print('Saved Best Model!')
        else:
            early_stopping_counter += 1
            print(f'No improvement. Early stopping counter: {early_stopping_counter}/{patience}')

            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        plot_metrics(metrics_logger.get_metrics_history(), config['output_dir'])
