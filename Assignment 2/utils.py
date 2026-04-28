import warnings
from typing import Optional, Callable, Tuple
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import f1_score


# ──────────────────────────────────────────────────────────────────────────────
# 1.  GradCAM core
# ──────────────────────────────────────────────────────────────────────────────


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: list = []

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, *_):
        self.remove_hooks()

    def _register_hooks(self):
        def _save_activation(_, __, output):
            self._activations = output.detach()

        def _save_gradient(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(_save_activation))
        self._handles.append(
            self.target_layer.register_full_backward_hook(_save_gradient)
        )

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        self.model.zero_grad()
        image_tensor = image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)  # forward pass

        # MULTI-LABEL FIX: Use sigmoid to find the most confident class if none provided
        if class_idx is None:
            class_idx = int(torch.sigmoid(logits).argmax(dim=1).item())

        # scalar score for the target class
        score = logits[0, class_idx]
        score.backward()  # backward pass

        # global average pooling of gradients  →  channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # normalise to [0, 1]
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, class_idx


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Helpers (PIL Swapped for CV2)
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError(
        "Could not auto-detect target layer. "
        "Pass `target_layer=model.your_last_conv_block` explicitly."
    )


def _denormalise(
    tensor: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    t = tensor.clone().cpu().squeeze(0)  # (C, H, W)
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).numpy()  # (H, W, C)
    t = np.clip(t, 0, 1)
    return (t * 255).astype(np.uint8)


def overlay_gradcam(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a GradCAM heatmap onto an image using PIL and Matplotlib."""
    img_np = _denormalise(image_tensor)  # (H, W, 3) uint8
    H, W = img_np.shape[:2]

    # Resize heatmap using PIL instead of cv2
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_u8).resize((W, H), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # Apply Jet Colormap using Matplotlib instead of cv2
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # Drops the alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.float32)

    overlay = (1 - alpha) * img_np.astype(np.float32) + alpha * heatmap_colored
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Inference + per-class F1 (Multi-Label Sigmoid Fix)
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_per_class_f1(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, list[torch.Tensor], list[list[int]], list[list[int]]]:
    model.eval()
    model.to(device)

    collected_images = []
    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)

        # MULTI-LABEL FIX: Sigmoid + 0.5 Threshold
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu().tolist()
        labels_list = labels.cpu().tolist()

        for img, lbl, pred in zip(images.cpu(), labels_list, preds):
            collected_images.append(img.unsqueeze(0))  # keep (1, C, H, W)
            all_labels.append(lbl)
            all_preds.append(pred)

    f1_per_class = f1_score(
        all_labels,
        all_preds,
        average=None,  # Returns an array of scores for all classes
        zero_division=0,
    )

    return f1_per_class, collected_images, all_labels, all_preds


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Single-image GradCAM visualisation
# ──────────────────────────────────────────────────────────────────────────────


def visualize_gradcam_single(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    target_class_idx: int,
    true_array: list[int],
    pred_array: list[int],
    target_layer: Optional[nn.Module] = None,
    ax_row: Optional[list] = None,
    title_prefix: str = "",
) -> None:
    if target_layer is None:
        target_layer = _resolve_target_layer(model)

    model.eval()
    model.to(device)
    img = image_tensor.to(device)

    with GradCAM(model, target_layer) as gcam:
        heatmap, _ = gcam(img, class_idx=target_class_idx)

    target_class_name = class_names[target_class_idx]
    predicted_it = pred_array[target_class_idx] == 1
    actually_has_it = true_array[target_class_idx] == 1

    is_correct = predicted_it == actually_has_it
    color = "green" if is_correct else "red"

    # --- Extract explicit TRUE prediction names ---
    true_classes = [class_names[i] for i, val in enumerate(true_array) if val == 1]
    if len(true_classes) == 0:
        true_title_text = "None"
    else:
        true_title_text = ", ".join(true_classes)
        # Truncate if it's too long
        if len(true_title_text) > 35:
            true_title_text = true_title_text[:32] + "..."

    # --- Extract explicit PRED prediction names ---
    if predicted_it:
        pred_title_text = "Correct Prediction"
    else:
        guessed_classes = [
            class_names[i] for i, val in enumerate(pred_array) if val == 1
        ]
        if len(guessed_classes) == 0:
            pred_title_text = "None"
        else:
            pred_title_text = ", ".join(guessed_classes)
            if len(pred_title_text) > 35:
                pred_title_text = pred_title_text[:32] + "..."

    img_orig = _denormalise(image_tensor)
    img_overlay = overlay_gradcam(image_tensor, heatmap)

    standalone = ax_row is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    else:
        axes = ax_row

    # LEFT IMAGE: Original Image with TRUE Labels
    axes[0].imshow(img_orig)
    axes[0].set_title(
        f"{title_prefix}True: {true_title_text}",
        fontsize=9,
        color="black",
    )
    axes[0].axis("off")

    # RIGHT IMAGE: Overlay Image with PRED Labels
    axes[1].imshow(img_overlay)
    axes[1].set_title(
        f"Pred: {pred_title_text}",
        fontsize=9,
        color=color,
        fontweight="bold",
    )
    axes[1].axis("off")

    if standalone:
        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Best / worst class panels (DIVERSE TIE-HANDLING)
# ──────────────────────────────────────────────────────────────────────────────


def _collect_samples_for_classes(
    class_indices: list[int],
    all_images: list,
    all_labels: list,
    all_preds: list,
    n_samples: int,
) -> list[tuple]:
    """Collects up to n_samples in a diverse round-robin fashion from multiple classes."""
    class_pools = {idx: [] for idx in class_indices}
    for img, lbl, pred in zip(all_images, all_labels, all_preds):
        for idx in class_indices:
            if lbl[idx] == 1:
                class_pools[idx].append((img, lbl, pred, idx))

    samples = []
    pool_keys = list(class_indices)
    pointers = {idx: 0 for idx in class_indices}

    while len(samples) < n_samples:
        added_this_round = False
        for idx in pool_keys:
            if len(samples) >= n_samples:
                break
            if pointers[idx] < len(class_pools[idx]):
                samples.append(class_pools[idx][pointers[idx]])
                pointers[idx] += 1
                added_this_round = True

        if not added_this_round:
            break

    return samples


def _render_class_panel(
    title: str,
    target_class_names: list[str],
    f1_score_val: float,
    samples: list[tuple],
    class_names: list[str],
    model: nn.Module,
    device: torch.device,
    target_layer: nn.Module,
    label_color: str = "black",
) -> None:
    n = len(samples)
    if n == 0:
        warnings.warn("No samples found for these classes. Skipping panel.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(7, 3.5 * n))
    if n == 1:
        axes = [axes]  # make iterable

    names_str = ", ".join(f"'{name}'" for name in target_class_names)
    if len(names_str) > 60:
        names_str = names_str[:57] + "..."

    fig.suptitle(
        f"{title}\nPer-class F1 = {f1_score_val:.3f}",
        fontsize=13,
        fontweight="bold",
        color=label_color,
        y=1.01,
    )

    for row_idx, (img_tensor, true_array, pred_array, target_idx) in enumerate(samples):
        specific_class_name = class_names[target_idx]
        visualize_gradcam_single(
            model=model,
            image_tensor=img_tensor,
            class_names=class_names,
            device=device,
            target_class_idx=target_idx,
            true_array=true_array,
            pred_array=pred_array,
            target_layer=target_layer,
            ax_row=axes[row_idx],
            title_prefix=f"Sample {row_idx + 1} -  ",
        )

    plt.tight_layout()
    plt.show()


def visualize_best_class(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    target_layer: Optional[nn.Module] = None,
    n_samples: int = 3,
) -> None:
    if target_layer is None:
        target_layer = _resolve_target_layer(model)

    num_classes = len(class_names)
    f1_scores, all_images, all_labels, all_preds = compute_per_class_f1(
        model, dataloader, device, num_classes
    )

    max_f1 = np.max(f1_scores)
    best_indices = np.where(f1_scores == max_f1)[0].tolist()
    best_names = [class_names[i] for i in best_indices]

    samples = _collect_samples_for_classes(
        best_indices, all_images, all_labels, all_preds, n_samples
    )

    print(
        f"[Best class(es)]  {', '.join(f'{n}' for n in best_names)} - F1 = {max_f1:.4f}"
    )
    _render_class_panel(
        title="Best Performing Class(es)",
        target_class_names=best_names,
        f1_score_val=max_f1,
        samples=samples,
        class_names=class_names,
        model=model,
        device=device,
        target_layer=target_layer,
        label_color="green",
    )


def visualize_worst_class(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    target_layer: Optional[nn.Module] = None,
    n_samples: int = 3,
) -> None:
    if target_layer is None:
        target_layer = _resolve_target_layer(model)

    num_classes = len(class_names)
    f1_scores, all_images, all_labels, all_preds = compute_per_class_f1(
        model, dataloader, device, num_classes
    )

    min_f1 = np.min(f1_scores)
    worst_indices = np.where(f1_scores == min_f1)[0].tolist()
    worst_names = [class_names[i] for i in worst_indices]

    samples = _collect_samples_for_classes(
        worst_indices, all_images, all_labels, all_preds, n_samples
    )

    print(
        f"[Worst class(es)]  {', '.join(f'{n}' for n in worst_names)} - F1 = {min_f1:.4f}"
    )
    _render_class_panel(
        title="Worst Performing Class(es)",
        target_class_names=worst_names,
        f1_score_val=min_f1,
        samples=samples,
        class_names=class_names,
        model=model,
        device=device,
        target_layer=target_layer,
        label_color="red",
    )


def visualize_best_and_worst(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    target_layer: Optional[nn.Module] = None,
    n_samples: int = 3,
) -> None:
    if target_layer is None:
        target_layer = _resolve_target_layer(model)

    num_classes = len(class_names)
    print("Computing per-class F1 scores ...")
    f1_scores, all_images, all_labels, all_preds = compute_per_class_f1(
        model, dataloader, device, num_classes
    )

    best_indices = np.where(f1_scores == np.max(f1_scores))[0].tolist()
    worst_indices = np.where(f1_scores == np.min(f1_scores))[0].tolist()

    # ── summary bar chart ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.3), 4))
    bar_colors = [
        (
            "#4caf50"
            if i in best_indices
            else "#f44336" if i in worst_indices else "#90caf9"
        )
        for i in range(num_classes)
    ]
    ax.bar(class_names, f1_scores, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score (per class)")
    ax.set_title("Per-Class F1 Scores", fontweight="bold")
    ax.tick_params(axis="x", rotation=90)
    best_patch = mpatches.Patch(color="#4caf50", label="Best class(es)")
    worst_patch = mpatches.Patch(color="#f44336", label="Worst class(es)")
    other_patch = mpatches.Patch(color="#90caf9", label="Other classes")
    ax.legend(handles=[best_patch, worst_patch, other_patch], fontsize=8)
    plt.tight_layout()
    plt.show()

    # ── best class panel ──────────────────────────────────────────────────────
    best_max_f1 = np.max(f1_scores)
    best_names = [class_names[i] for i in best_indices]
    print(
        f"\n[Best class(es)]  {', '.join(f'{n}' for n in best_names)} - F1 = {best_max_f1:.4f}"
    )

    best_samples = _collect_samples_for_classes(
        best_indices, all_images, all_labels, all_preds, n_samples
    )

    _render_class_panel(
        title="Best Performing Class(es)",
        target_class_names=best_names,
        f1_score_val=best_max_f1,
        samples=best_samples,
        class_names=class_names,
        model=model,
        device=device,
        target_layer=target_layer,
        label_color="green",
    )

    # ── worst class panel ─────────────────────────────────────────────────────
    worst_min_f1 = np.min(f1_scores)
    worst_names = [class_names[i] for i in worst_indices]
    print(
        f"\n[Worst class(es)]  {', '.join(f'{n}' for n in worst_names)} - F1 = {worst_min_f1:.4f}"
    )

    worst_samples = _collect_samples_for_classes(
        worst_indices, all_images, all_labels, all_preds, n_samples
    )

    _render_class_panel(
        title="Worst Performing Class(es)",
        target_class_names=worst_names,
        f1_score_val=worst_min_f1,
        samples=worst_samples,
        class_names=class_names,
        model=model,
        device=device,
        target_layer=target_layer,
        label_color="red",
    )


class LegoDataset(Dataset):
    """
    A custom PyTorch Dataset for loading Lego images and their associated labels.

    This dataset assumes a CSV file where the first column contains image filenames
    and the subsequent columns contain multi-label classification targets.

    Attributes:
        df (pd.DataFrame): The underlying dataframe containing paths and labels.
        img_dir (str): The base directory where images are stored.
        transform (Optional[Callable]): A function/transform to apply to the images.
        img_paths (np.ndarray): Array of filenames extracted from the CSV.
        labels (np.ndarray): Array of float32 labels extracted from the CSV.
    """

    def __init__(
        self, csv_file: str, img_dir: str = ".", transform: Optional[Callable] = None
    ) -> None:
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.ToTensor()

        # paths are in the first column, multilabel targets in the rest
        self.img_paths = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1:].values.astype("float32")

    def __len__(self) -> int:
        """
        Returns the number of Samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches a single image and its corresponding labels.

        Args:
            idx: Index of the sample to fetch.

        Returns:
            A tuple containing (image, label), where image is a tensor
            (after transform) and label is a float32 tensor.
        """

        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx])
        image = self.transform(image)

        return image, label


def get_train_transforms(image_size=224):
    """
    Returns the data augmentation and normalization pipeline
    for the training dataset.
    """
    return v2.Compose(
        [
            v2.ToImage(),  # convert input to torchvision image format
            v2.ToDtype(torch.float32, scale=True),  # scale pixel values to [0, 1]
            # --- Augmentations ---
            v2.RandomResizedCrop(
                size=image_size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.NEAREST,
            ),
            v2.RandomRotation(
                degrees=15, interpolation=InterpolationMode.NEAREST, fill=1
            ),
            v2.RandomPerspective(
                distortion_scale=0.3,
                p=0.2,
                interpolation=InterpolationMode.NEAREST,
                fill=1,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            # --- Normalization (ImageNet Stats) ---
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_valid_transforms(image_size=224):
    """
    Returns the strict resizing and normalization pipeline
    for the validation/test datasets (NO augmentations).
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(
                size=(image_size, image_size), interpolation=InterpolationMode.NEAREST
            ),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )

        # Calculate probabilities
        pt = torch.exp(-bce_loss)

        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def find_best_threshold(outputs, targets):
    thresholds = np.linspace(0.1, 0.9, 20)
    best_thresh, best_f1 = 0.5, 0

    for t in thresholds:
        preds = (outputs > t).astype(int)
        f1 = f1_score(targets, preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1