import time
from pathlib import Path
import cv2
import torch
import numpy as np
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
    """Computes all-pairs intersection areas using broadcasting."""
    b1 = boxes.unsqueeze(1)
    b2 = boxes.unsqueeze(0)
    xA = torch.max(b1[..., 0], b2[..., 0])
    yA = torch.max(b1[..., 1], b2[..., 1])
    xB = torch.min(b1[..., 2], b2[..., 2])
    yB = torch.min(b1[..., 3], b2[..., 3])
    inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    mask = torch.triu(torch.ones_like(inter_area), diagonal=1).bool()
    return inter_area[mask]


def compute_bboxes_area_loss_vectorized(boxes_x1, mask_tensor, top_k=100):
    """
    mask_tensor: [640, 640] boolean tensor of the specific object zone
    """
    if boxes_x1.shape[0] > top_k:
        boxes_x1 = boxes_x1[:top_k]
    if len(boxes_x1) < 2 or not mask_tensor.any():
        return torch.tensor(0.0, device=boxes_x1.device, requires_grad=True)

    pos = torch.where(mask_tensor)
    z_ymin, z_xmin = pos[0].min(), pos[1].min()
    z_ymax, z_xmax = pos[0].max(), pos[1].max()

    width = (z_xmax - z_xmin).float() + 1e-6
    height = (z_ymax - z_ymin).float() + 1e-6

    d1_zone = width + height
    d2_zone = (width ** 2 + height ** 2) ** 0.5
    zone_area = (width * height) + 1e-6  # α·β bounding rectangle area (Eq. 11)

    l1_loss = get_vectorized_distances(boxes_x1, p=1).mean() / d1_zone
    l2_loss = get_vectorized_distances(boxes_x1, p=2).mean() / d2_zone

    inter_areas = get_vectorized_intersections(boxes_x1)
    inter_loss = inter_areas.max() / zone_area

    return (l1_loss + l2_loss + inter_loss) / 3


def compute_max_objects_loss(output_patch, target_class=0, conf_thres=0.25):
    # Compute class-weighted confidence scores: (batch, anchors, num_classes)
    x2 = output_patch[:, :, 5:] * output_patch[:, :, 4:5]

    conf, j = x2.max(2, keepdim=False)
    all_target_conf = x2[:, :, target_class]
    under_thr_target_conf = all_target_conf[conf < conf_thres]

    # Fix: normalize by total number of predictions (|Cb|), not just batch size (Eq. 4)
    total_preds = conf.numel()
    conf_avg = (conf.view(-1) > conf_thres).sum().item() / total_preds

    # Guard against empty candidate set — avoids autograd crash on zero-element tensors
    if under_thr_target_conf.numel() == 0:
        return torch.tensor(0.0, device=output_patch.device, requires_grad=True), conf_avg

    # Eq. 3: conf(B) = max(Tconf - Bconf, 0) — no gradient incentive above threshold
    x3 = torch.clamp(-under_thr_target_conf + conf_thres, min=0)

    # Eq. 4: max_obj = (1 / |Cb|) * sum over Ca of conf(B)
    mean_conf = x3.sum() / total_preds

    return mean_conf, conf_avg


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
        self.patch_conf_threshold = 0.001
        self.time = []

        # Load YOLOv5 model directly from torch hub
        self.model = (
            torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
            .to(self.device)
            .eval()
        )
        print(f"[Model] Loaded on {self.device}")

    def loss_func(self, loss_hypers, target_class, output_patch, mask_tensors=None):
        """
        Full loss from Eq. 12:
            L = lambda_1 * max_obj + lambda_2 * (l1 + l2 + inter) / 3
        """
        # --- Term 1: max_obj loss (Eq. 3 & 4) ---
        max_objects_loss, pass_to_NMS = compute_max_objects_loss(output_patch, target_class,
                                                                 conf_thres=self.conf_threshold)
        loss = max_objects_loss * loss_hypers['lambda_1']

        # --- Term 2: spatial spread loss (Eq. 6, 8, 11) ---
        if loss_hypers.get('lambda_2a', 0) > 0 and mask_tensors is not None:
            # Extract candidate boxes (Cb): all predictions above patch_conf_threshold
            # output_patch shape: (batch, 25200, 85) — [cx, cy, w, h, obj_conf, cls...]
            x2 = output_patch[:, :, 5:] * output_patch[:, :, 4:5]
            conf, _ = x2.max(2)  # (batch, 25200)

            # Gather boxes above patch confidence threshold across the whole batch
            above_mask = conf > self.patch_conf_threshold  # (batch, 25200)
            raw_boxes = output_patch[:, :, :4]  # cx, cy, w, h

            # Convert cx,cy,w,h → x1,y1,x2,y2 for all predictions, then filter
            cx = raw_boxes[..., 0]
            cy = raw_boxes[..., 1]
            w  = raw_boxes[..., 2]
            h  = raw_boxes[..., 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2_coord = cx + w / 2
            y2_coord = cy + h / 2
            boxes_xyxy = torch.stack([x1, y1, x2_coord, y2_coord], dim=-1)  # (batch, 25200, 4)

            # Flatten across batch and keep only above-threshold candidates
            boxes_flat = boxes_xyxy.view(-1, 4)
            mask_flat  = above_mask.view(-1)
            candidate_boxes = boxes_flat[mask_flat]  # (N_candidates, 4)

            # Use the first mask in the batch as the zone reference
            zone_mask = mask_tensors[0].to(self.device)

            bbox_area_loss = compute_bboxes_area_loss_vectorized(candidate_boxes, zone_mask)
            loss = loss + bbox_area_loss * loss_hypers['lambda_2a']

        return loss

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

        data_iter = iter(dataloader)

        for curr_iter in range(1, self.max_iter + 1):
            self.time.append(time.time() - start)

            # Fetch next mini-batch, resetting iterator if it runs out
            try:
                victim_imgs, mask_tensors = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                victim_imgs, mask_tensors = next(data_iter)

            victim_imgs = victim_imgs.to(self.device)
            mask_tensors = mask_tensors.to(self.device)

            # 2. Add patch locally and clamp
            applied_patch = torch.clamp(victim_imgs + adv_patch, 0.0, 1.0)

            # 3. Forward pass through YOLO
            output_patch = self.model(applied_patch)[0]  # (batch, 25200, 85)

            # 4. Calculate Loss (Eq. 12: max_obj + spatial spread terms)
            loss = self.loss_func(loss_hypers, target_class, output_patch, mask_tensors=mask_tensors)

            # --------------------------------------------------------- #
            # EXPLICIT GRADIENT CALCULATION & UPDATE
            # --------------------------------------------------------- #
            data_grad = torch.autograd.grad(outputs=loss, inputs=adv_patch)[0]

            # FGSM Update Step
            tmp_adv_patch = adv_patch.detach() - fgsm_epsilon * data_grad.sign()

            # --------------------------------------------------------- #
            # PROJECTION
            # --------------------------------------------------------- #
            perturbation = tmp_adv_patch - base_patch
            norm = torch.sqrt(torch.sum(torch.square(perturbation)))

            factor = min(1.0, pgd_epsilon / norm.item()) if norm.item() > 0 else 1.0
            perturbation = perturbation * factor

            with torch.no_grad():
                adv_patch = torch.clamp(base_patch + perturbation, 0.0, 1.0)
            adv_patch = adv_patch.detach().requires_grad_(True)

            # SAVE Intermediate Progress
            if curr_iter % 10 == 0:
                img_save_path = save_patch_dir / f'iter={curr_iter}.PNG'
                transforms.ToPILImage()(adv_patch.detach().cpu()).save(img_save_path)

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
    MINI_BATCH_SIZE = 8  # images per mini-batch (tune to fit VRAM)
    ITERATIONS = 500  # patch update steps per class
    EPSILON = 0.005  # FGSM step size

    GLOBAL_PATCH_PATH = Path('global_patch_checkpoint.pt')

    image_directories = [
        r'C:\Users\rsvis\UTD\Sem 6\Data and applications security\Artifact eval\Repo\DetStorm\out_segments\1775285345',
        r'C:\Users\rsvis\UTD\Sem 6\Data and applications security\Artifact eval\Repo\DetStorm\out_segments\1776223121',
    ]

    classes_important = {
        'bridge', 'building', 'bus', 'car', 'ceiling', 'column', 'earth',
        'floor', 'fence', 'grass', 'house', 'person', 'pole', 'road',
        'rock', 'runway', 'sidewalk', 'signboard', 'traffic', 'truck', 'van',
    }

    # ── Resume / skip logic ───────────────────────────────────────────────
    skip_until_class = None
    found_skip = False

    global_patch = None
    if GLOBAL_PATCH_PATH.exists():
        global_patch = torch.load(GLOBAL_PATCH_PATH)
        print(f"[Resume] Loaded global patch from {GLOBAL_PATCH_PATH}")
    else:
        print("[Start] No global patch found — cold start.")

    # ── Main loop ─────────────────────────────────────────────────────────
    for root_dir in image_directories:
        # Avoid error if path structure is incorrect
        if not os.path.exists(root_dir):
            print(f"[Warning] Directory {root_dir} not found. Skipping.")
            continue

        # Reset skip state per directory so resume logic applies to each root independently
        found_skip = (skip_until_class is None)

        class_folders = sorted(glob.glob(os.path.join(root_dir, '*/')))

        for cls_path in class_folders:
            folder_name = Path(cls_path).name

            # Skip until the target class is reached
            if skip_until_class is not None and not found_skip:
                if folder_name == skip_until_class:
                    print(f"[Skip] Resuming from class: {folder_name}")
                    found_skip = True
                else:
                    continue

            print(f"\n[Processing] Class: {folder_name}")
            is_important = folder_name in classes_important

            if not is_important:
                print(f"  [Skip] Class '{folder_name}' is not in classes_important whitelist.")
                continue

            img_paths = sorted(glob.glob(os.path.join(cls_path, '*.jpg')))
            if not img_paths:
                print(f"  [Skip] No .jpg images found in {folder_name}.")
                continue

            print(f"  Found {len(img_paths)} images | Starting optimization...")

            global_patch = run_attack(
                max_iter=ITERATIONS,
                fgsm_epsilon=EPSILON,
                img_paths=img_paths,
                folder_name=folder_name,
                mini_batch_size=MINI_BATCH_SIZE,
                warm_start_patch=global_patch,
            )

            # Persist updated patch for crash recovery
            torch.save(global_patch, GLOBAL_PATCH_PATH)
            print(f"[Checkpoint] Global patch saved → {GLOBAL_PATCH_PATH}")

    print("\n[Done] Pipeline execution complete.")