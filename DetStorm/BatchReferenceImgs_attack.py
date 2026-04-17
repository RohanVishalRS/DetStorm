import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
import math
import gc
from PIL import Image


# ---------------------------------------------------------------------------
# Vectorized Helper Functions
# ---------------------------------------------------------------------------

def get_vectorized_distances(boxes, p=2):
    """Computes all-pairs distances between box centroids."""
    centroids = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                             (boxes[:, 1] + boxes[:, 3]) / 2], dim=1)
    dists = torch.cdist(centroids, centroids, p=p)
    mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
    return dists[mask]


def get_vectorized_intersections(boxes):
    """Follows paper Eq 9-10: measures max-extent overflow between box pairs."""
    b1 = boxes.unsqueeze(1)  # (N, 1, 4)
    b2 = boxes.unsqueeze(0)  # (1, N, 4)

    # Eq 9: max extents of each box on each axis
    # For xyxy format: x2 > x1, y2 > y1, so max is just x2, y2
    Xa = torch.max(b1[..., 0], b1[..., 2])  # max x of Bi = x2 of Bi
    Xb = torch.max(b2[..., 0], b2[..., 2])  # max x of Bj = x2 of Bj
    Ya = torch.max(b1[..., 1], b1[..., 3])  # max y of Bi = y2 of Bi
    Yb = torch.max(b2[..., 1], b2[..., 3])  # max y of Bj = y2 of Bj

    # Eq 10
    inter_area = torch.abs(
        torch.clamp(Xb - Xa, min=0) * torch.clamp(Yb - Ya, min=0)
    )

    mask = torch.triu(torch.ones(inter_area.shape, device=boxes.device), diagonal=1).bool()
    return inter_area[mask]


def compute_bboxes_area_loss_vectorized(boxes_x1, mask_tensor, top_k=100):
    if boxes_x1.shape[0] > top_k:
        boxes_x1 = boxes_x1[:top_k]

    zero = boxes_x1.sum() * 0.0

    if boxes_x1.shape[0] < 2 or not mask_tensor.any():
        return zero

    pos = torch.where(mask_tensor)
    z_ymin, z_xmin = pos[0].min(), pos[1].min()
    z_ymax, z_xmax = pos[0].max(), pos[1].max()

    width  = (z_xmax - z_xmin).float()
    height = (z_ymax - z_ymin).float()

    d1_zone = width + height
    d2_zone = (width ** 2 + height ** 2) ** 0.5
    zone_area = width * height

    l1_loss = get_vectorized_distances(boxes_x1, p=1).mean() / d1_zone
    l2_loss = get_vectorized_distances(boxes_x1, p=2).mean() / d2_zone

    inter_areas = get_vectorized_intersections(boxes_x1)
    inter_loss  = torch.clamp(inter_areas.max() / zone_area, max=1.0)

    return (l1_loss + l2_loss + inter_loss) / 3


def compute_max_objects_loss(zone_target_conf, conf_thres=0.25):
    """
    Computes max_objects_loss (Eq. 3 + Eq. 4) for a single image's zone-filtered
    target-class confidences.

    Args:
        zone_target_conf: 1-D tensor of target-class confidence scores for all
                          anchors whose centroid falls inside the zone (|Cb| entries).
        conf_thres:       Confidence threshold Tconf.

    Returns:
        loss:      Scalar loss tensor (differentiable).
        above_count: Number of anchors already above the threshold (int).
    """
    total_preds_zone = zone_target_conf.numel()
    above_count = (zone_target_conf >= conf_thres).sum().item()

    if total_preds_zone == 0:
        return torch.tensor(0.0, device=zone_target_conf.device, requires_grad=True), above_count

    under_thr_mask = zone_target_conf < conf_thres
    under_thr_conf = zone_target_conf[under_thr_mask]

    if under_thr_conf.numel() == 0:
        return zone_target_conf.sum() * 0.0, above_count  # zero with grad

    # Eq. 3 + Eq. 4
    x3 = torch.clamp(conf_thres - under_thr_conf, min=0)
    loss = x3.sum() / total_preds_zone

    return loss, above_count


# ---------------------------------------------------------------------------
# Lazy-loading Dataset (Reads from disk per mini-batch)
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = img_path.replace(".jpg", ".npy")

        try:
            with Image.open(img_path).convert('RGB') as img:
                img_resized = img.resize((640, 640))
                arr = np.array(img_resized).transpose(2, 0, 1)
                img_tensor = torch.from_numpy(arr).float() / 255.0

            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                mask_resized = cv2.resize(mask.astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.from_numpy(mask_resized).bool()
            else:
                mask_tensor = torch.ones(640, 640).bool()  # Fallback entire image if mask missing

            return img_tensor, mask_tensor
        except Exception as e:
            print(f"[Warning] Could not load {img_path}: {e}")
            return torch.zeros(3, 640, 640), torch.ones(640, 640).bool()


# ---------------------------------------------------------------------------
# Core Adversarial Patch Generator
# ---------------------------------------------------------------------------

class PhantomGenerator:
    def __init__(self, model_name, model_ckpt_path, max_iter):
        self.max_iter = max_iter
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.patch_size_H = 640
        self.patch_size_W = 640

        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.time = []

        # Load YOLOv5 model directly from torch hub
        self.model = (
            torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
            .to(self.device)
            .eval()
        )
        print(f"[Model] Loaded on {self.device}")

    def loss_func(self, loss_hypers, target_class, output_patch, mask_tensors=None):
        x2 = output_patch[:, :, 5:] * output_patch[:, :, 4:5]
        conf, _ = x2.max(2)
        cx = output_patch[..., 0]
        cy = output_patch[..., 1]
        w = output_patch[..., 2]
        h = output_patch[..., 3]
        boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        zero = output_patch.sum() * 0.0
        max_objects_loss_total = zero
        bbox_area_loss = zero
        valid_count = 0
        total_preds_all = 0
        pass_to_NMS_total = 0

        for i in range(output_patch.shape[0]):
            zone_mask_i = mask_tensors[i].to(self.device)

            if not zone_mask_i.any():
                continue

            # --- Get zone bounding box in PIXEL coords ---
            pos = torch.where(zone_mask_i)
            z_ymin, z_xmin = pos[0].min().float(), pos[1].min().float()
            z_ymax, z_xmax = pos[0].max().float(), pos[1].max().float()

            # --- Normalize to [0, 1] to match YOLO anchor coords ---
            box_cx = (boxes_xyxy[i, :, 0] + boxes_xyxy[i, :, 2]) / 2
            box_cy = (boxes_xyxy[i, :, 1] + boxes_xyxy[i, :, 3]) / 2
            in_zone = (
                    (box_cx >= z_xmin) & (box_cx <= z_xmax) &
                    (box_cy >= z_ymin) & (box_cy <= z_ymax)
            )

            # --- Term 1: max_objects_loss restricted to zone ---
            zone_target_conf = x2[i, :, target_class][in_zone]
            total_preds_all += zone_target_conf.numel()

            img_obj_loss, above_count = compute_max_objects_loss(zone_target_conf, conf_thres=self.conf_threshold)
            pass_to_NMS_total += above_count
            max_objects_loss_total = max_objects_loss_total + img_obj_loss

            # --- Term 2: spatial loss restricted to zone ---
            if loss_hypers.get('lambda_2a', 0) > 0 and mask_tensors is not None:
                above_mask_i = conf[i] > self.conf_threshold
                in_zone_above = in_zone & above_mask_i
                candidate_boxes_i = boxes_xyxy[i][in_zone_above]

                img_loss = compute_bboxes_area_loss_vectorized(candidate_boxes_i, zone_mask_i, top_k=2000)
                if not torch.isnan(img_loss):
                    bbox_area_loss = bbox_area_loss + img_loss
                    valid_count += 1

        n_images = output_patch.shape[0]
        max_objects_loss = max_objects_loss_total / n_images
        l1_l2_IoU_loss = (bbox_area_loss / valid_count) if valid_count > 0 else zero
        pass_to_NMS = pass_to_NMS_total / n_images

        return max_objects_loss, l1_l2_IoU_loss, pass_to_NMS

    def generate_phantom(self, dataloader, loss_hypers, target_class, pgd_epsilon, fgsm_epsilon, folder_name=None,
                         warm_start_patch=None):
        start = time.time()

        # 1. Initialize the base patch
        if warm_start_patch is not None:
            base_patch = warm_start_patch.clone().detach().to(self.device)
        else:
            base_patch = torch.zeros([3, self.patch_size_H, self.patch_size_W], device=self.device)

        # adv_patch is what we actually optimize
        adv_patch = base_patch.clone().detach()
        adv_patch.requires_grad_(True)

        final_dir = Path(str(folder_name)) if folder_name else Path('output_patch')
        final_dir.mkdir(parents=True, exist_ok=True)
        save_patch_dir = final_dir / 'save_patch'
        save_patch_dir.mkdir(parents=True, exist_ok=True)

        for curr_iter in range(1, self.max_iter + 1):
            self.time.append(time.time() - start)

            # --------------------------------------------------------- #
            # ACCUMULATE GRADIENTS OVER THE FULL DATALOADER (one epoch)
            # --------------------------------------------------------- #
            accumulated_grad = None
            epoch_loss = 0.0
            epoch_max_obj_loss = 0.0
            epoch_dist_loss = 0.0
            epoch_pass_to_NMS = 0.0
            epoch_total_images = 0
            num_batches = 0
            last_output_patch = None

            for victim_imgs, mask_tensors in dataloader:
                victim_imgs = victim_imgs.to(self.device)
                mask_tensors = mask_tensors.to(self.device)

                # 2. Add patch locally and clamp
                mask_3c = mask_tensors.float().unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, H, W)
                applied_patch = torch.clamp(victim_imgs + adv_patch * mask_3c, 0.0, 1.0)

                # 3. Forward pass through YOLO
                output_patch = self.model(applied_patch)[0]  # (batch, 25200, 85)
                last_output_patch = output_patch

                # 4. Calculate Loss (Eq. 12: max_obj + spatial spread terms)
                max_objects_loss, l1_l2_IoU_loss, pass_to_NMS = self.loss_func(
                    loss_hypers, target_class, output_patch, mask_tensors=mask_tensors
                )
                loss = max_objects_loss * loss_hypers['lambda_1'] + l1_l2_IoU_loss * loss_hypers['lambda_2a']

                if not loss.requires_grad:
                    continue

                # Accumulate gradient for this mini-batch
                batch_grad = torch.autograd.grad(outputs=loss, inputs=adv_patch, allow_unused=True)[0]
                if batch_grad is not None:
                    accumulated_grad = batch_grad if accumulated_grad is None else accumulated_grad + batch_grad

                batch_size = victim_imgs.shape[0]
                # Accumulate totals (loss_func returns per-image averages, so scale back up)
                epoch_loss += loss.item() * batch_size
                epoch_max_obj_loss += max_objects_loss.item() * batch_size
                epoch_dist_loss += l1_l2_IoU_loss.item() * batch_size
                epoch_pass_to_NMS += pass_to_NMS * batch_size  # pass_to_NMS is per-image avg
                epoch_total_images += batch_size
                num_batches += 1

            if accumulated_grad is None or num_batches == 0:
                print(f"[Iter {curr_iter}] No valid gradients this epoch; skipping update")
                continue

            # Average the accumulated gradient over all mini-batches
            avg_grad = accumulated_grad / num_batches

            # --------------------------------------------------------- #
            # EXPLICIT GRADIENT CALCULATION & UPDATE  (once per epoch)
            # --------------------------------------------------------- #

            # FGSM Update Step:
            # Per the document: x' = x + ε·sign(∇_x J)
            # Here adv_patch plays the role of x (the full adversarial image/noise),
            # and avg_grad = ∂loss/∂adv_patch flows through applied_patch = clamp(img + adv_patch * mask).
            # We SUBTRACT because we are minimizing the loss (maximizing detections).
            tmp_adv_patch = adv_patch.detach() - fgsm_epsilon * avg_grad.sign()

            # --------------------------------------------------------- #
            # PROJECTION  (L∞ ball around base_patch, per BIM/PGD paper)
            # --------------------------------------------------------- #
            # Clamp the per-element perturbation to [-pgd_epsilon, +pgd_epsilon]
            # then clamp the resulting patch to valid pixel range [0, 1].
            perturbation = torch.clamp(tmp_adv_patch - base_patch, -pgd_epsilon, pgd_epsilon)

            with torch.no_grad():
                adv_patch = torch.clamp(base_patch + perturbation, 0.0, 1.0)
            adv_patch = adv_patch.detach().requires_grad_(True)

            # SAVE Intermediate Progress
            if curr_iter % 10 == 0:
                img_save_path = save_patch_dir / f'iter={curr_iter}.PNG'
                transforms.ToPILImage()(adv_patch.detach().cpu()).save(img_save_path)
                # All metrics are per-image averages across the full epoch
                avg_loss = epoch_loss / epoch_total_images
                avg_max_obj = epoch_max_obj_loss / epoch_total_images
                avg_dist = epoch_dist_loss / epoch_total_images
                avg_candidates = epoch_pass_to_NMS / epoch_total_images  # avg candidate boxes per image
                print(f"[Iter {curr_iter:4d}] loss={avg_loss:.6f} | avg candidate boxes/img={avg_candidates:.2f} | Obj loss={avg_max_obj:.6f} | Dist loss={avg_dist:.6f}")

        # --------------------------------------------------------- #
        # FINAL EXPORT
        # --------------------------------------------------------- #
        print(f"\n[Info] Maximum iterations reached. Saving final patch to: {final_dir}")

        final_img_path = final_dir / 'final_patch.PNG'
        transforms.ToPILImage()(adv_patch.detach().cpu()).save(final_img_path)

        final_npy_path = final_dir / 'final_patch.npy'
        final_patch_array = adv_patch.detach().cpu().numpy()
        np.save(str(final_npy_path), final_patch_array)

        print(f"[Success] Saved {final_img_path.name} and {final_npy_path.name}")

        # Return state for warm-starts
        return adv_patch.detach().cpu()


# ---------------------------------------------------------------------------
# Attack Runner
# ---------------------------------------------------------------------------

def run_attack(max_iter, fgsm_epsilon, img_paths, folder_name,
               mini_batch_size=8, warm_start_patch=None):
    loss_hypers = {
        'lambda_1': 1,
        'lambda_2a': 5,
    }

    dataset = ImageFolderDataset(img_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=mini_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,  # Best to leave at 0 on Windows to avoid DataLoader multiprocessing freezes
        pin_memory=False,  # pin_memory has no effect with num_workers=0
    )

    generator = PhantomGenerator(
        model_name='yolov5',
        model_ckpt_path='yolov5s.pt',
        max_iter=max_iter,
    )

    final_patch = generator.generate_phantom(
        dataloader=dataloader,
        loss_hypers=loss_hypers,
        target_class=0,  # Defaults to 'person', adjust as needed per folder
        pgd_epsilon=70.0,
        fgsm_epsilon=fgsm_epsilon,
        folder_name=folder_name,
        warm_start_patch=warm_start_patch,
    )

    # Clean up to keep VRAM strictly bounded
    del generator, dataloader, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return final_patch


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ── Configuration ─────────────────────────────────────────────────────
    MINI_BATCH_SIZE   = 10
    ITERATIONS        = 500
    EPSILON           = 0.05

    OUTPUT_ROOT       = Path('generated_patches')   # ← all patches saved here
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    image_directories = [
        r'\DetStorm\out_segments\1775285345',
        r'\DetStorm\out_segments\1776223121',
    ]

    classes_important = {
        'bridge', 'building', 'bus', 'car', 'person', 'pole', 'road', 'sidewalk', 'signboard', 'traffic', 'truck', 'van'
    }

    # ── Resume logic ──────────────────────────────────────────────────────
    skip_until_class = None   # Set to a class name string to resume, e.g. 'car'
    found_skip       = (skip_until_class is None)

    # ── Build per-class image list across ALL directories ─────────────────
    class_to_imgs = {}

    for root_dir in image_directories:
        if not os.path.exists(root_dir):
            print(f"[Warning] Directory {root_dir} not found. Skipping.")
            continue

        for cls_path in sorted(glob.glob(os.path.join(root_dir, '*/'))):
            folder_name = Path(cls_path).name

            if folder_name not in classes_important:
                continue

            img_paths = sorted(glob.glob(os.path.join(cls_path, '*.jpg')))
            if not img_paths:
                continue

            if folder_name not in class_to_imgs:
                class_to_imgs[folder_name] = []
            class_to_imgs[folder_name].extend(img_paths)

    print(f"[Info] Found {len(class_to_imgs)} classes across all directories:")
    for cls, imgs in sorted(class_to_imgs.items()):
        print(f"  {cls}: {len(imgs)} images")

    # ── Main loop — one independent patch per class ────────────────────────
    for folder_name, img_paths in sorted(class_to_imgs.items()):

        # Resume logic
        if not found_skip:
            if folder_name == skip_until_class:
                print(f"[Resume] Resuming from class: {folder_name}")
                found_skip = True
            else:
                print(f"[Skip] Skipping class: {folder_name}")
                continue

        # Output folder: generated_patches/<class_name>/
        class_output_dir = OUTPUT_ROOT / folder_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Processing] Class: {folder_name} | {len(img_paths)} images | Output: {class_output_dir}")

        class_patch = run_attack(
            max_iter=ITERATIONS,
            fgsm_epsilon=EPSILON,
            img_paths=img_paths,
            folder_name=str(class_output_dir),   # ← pass class output dir
            mini_batch_size=MINI_BATCH_SIZE,
            warm_start_patch=None,
        )

        # Save final patch as .pt and .PNG under generated_patches/<class_name>/
        torch.save(class_patch, class_output_dir / 'final_patch.pt')
        transforms.ToPILImage()(class_patch.cpu()).save(class_output_dir / 'final_patch.PNG')
        print(f"[Saved] {class_output_dir / 'final_patch.PNG'} + final_patch.pt")

        gc.collect()
        torch.cuda.empty_cache()

    print("\n[Done] Pipeline execution complete.")