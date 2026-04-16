import time
from pathlib import Path
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


def compute_bboxes_area_loss_vectorized(boxes_x1, patch_size, top_k=100):
    """BBox spread/overlap loss, capped at top_k boxes to prevent VRAM hang."""
    if boxes_x1.shape[0] > top_k:
        boxes_x1 = boxes_x1[:top_k]
    if len(boxes_x1) < 2:
        if len(boxes_x1) == 1:
            wh = boxes_x1[:, 2:] - boxes_x1[:, :2]
            return (wh[:, 0] * wh[:, 1]).mean() / (patch_size[0] * patch_size[1])
        return torch.tensor(0.0, device=boxes_x1.device, requires_grad=True)
    l1_dists  = get_vectorized_distances(boxes_x1, p=1)
    l2_dists  = get_vectorized_distances(boxes_x1, p=2)
    inter_areas = get_vectorized_intersections(boxes_x1)
    l1_loss   = l1_dists.mean()   / (patch_size[0] + patch_size[1])
    l2_loss   = l2_dists.mean()   / math.sqrt(patch_size[0]**2 + patch_size[1]**2)
    inter_loss = inter_areas.max() / (patch_size[0] * patch_size[1])
    return (l1_loss + l2_loss + inter_loss) / 3


# ---------------------------------------------------------------------------
# Lazy-loading Dataset  (no RAM spike — reads from disk per mini-batch)
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """
    Reads images from disk on demand.
    Only one mini-batch worth of images lives in RAM at any time.
    """
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.img_paths[idx]).convert('RGB') as img:
                img_resized = img.resize((640, 640))
                arr = np.array(img_resized).transpose(2, 0, 1)   # [3, H, W]
                return torch.from_numpy(arr).float() / 255.0
        except Exception as e:
            print(f"[Warning] Could not load {self.img_paths[idx]}: {e}")
            return torch.zeros(3, 640, 640)                        # safe fallback


# ---------------------------------------------------------------------------
# Core Adversarial Patch Generator
# ---------------------------------------------------------------------------

class PhantomGenerator:
    def __init__(self, model_name, model_ckpt_path, max_iter):
        self.max_iter  = max_iter
        self.device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.patch_size = (640, 640)

        # Load model (replace with your custom loader if needed)
        self.model = (
            torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
            .to(self.device)
            .eval()
        )
        print(f"[Model] Loaded on {self.device}")

    def generate_phantom(self, dataloader, loss_hypers, pgd_epsilon,
                         fgsm_epsilon, folder_name, warm_start_patch=None):
        """
        True mini-batch gradient descent:
          - Each *iteration* = one full pass over all mini-batches (one epoch).
          - Gradients are accumulated across all mini-batches.
          - ONE patch update is performed at the end of each iteration.
        """
        # ── Patch initialisation ──────────────────────────────────────────
        if warm_start_patch is not None:
            print("[Warm Start] Reusing patch from previous class.")
            adv_patch = warm_start_patch.to(self.device).detach().clone().requires_grad_(True)
        else:
            print("[Cold Start] Initialising patch from zeros.")
            adv_patch = torch.zeros(3, 640, 640, device=self.device, requires_grad=True)

        # ── Output directories ────────────────────────────────────────────
        save_path    = Path(folder_name)
        iter_patch_dir = save_path / 'iter_patches'
        save_path.mkdir(parents=True, exist_ok=True)
        iter_patch_dir.mkdir(parents=True, exist_ok=True)

        torch.autograd.set_detect_anomaly(False)

        num_batches = len(dataloader)
        conf_thres  = 0.25
        target_class = 0

        print(f"[Attack] {self.max_iter} iterations | {num_batches} mini-batches/iter "
              f"| {len(dataloader.dataset)} total images")

        for curr_iter in range(self.max_iter):
            epoch_loss    = 0.0
            last_num_boxes = 0

            # Zero patch gradient before accumulating over mini-batches
            if adv_patch.grad is not None:
                adv_patch.grad.zero_()

            # ── Mini-batch loop ───────────────────────────────────────────
            for batch_imgs in dataloader:
                batch_imgs = batch_imgs.to(self.device)           # [B, 3, 640, 640]
                applied    = torch.clamp(batch_imgs + adv_patch, 0.0, 1.0)

                outputs = self.model(applied)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]                          # [B, 25200, 85]

                # 1. Objectness / confidence loss
                class_confs = outputs[:, :, 4] * outputs[:, :, 5 + target_class]
                max_obj_loss = torch.mean(torch.clamp(conf_thres - class_confs, min=0))

                # 2. BBox area / spread loss
                area_losses = []
                for i in range(outputs.shape[0]):
                    img_out = outputs[i]
                    obj_mask = img_out[:, 4] > conf_thres
                    boxes    = img_out[obj_mask]
                    last_num_boxes = len(boxes)
                    if len(boxes) > 0:
                        _, idx = torch.sort(boxes[:, 4], descending=True)
                        area_losses.append(
                            compute_bboxes_area_loss_vectorized(
                                boxes[idx][:, :4], self.patch_size, top_k=2000
                            )
                        )

                bbox_loss = (torch.stack(area_losses).mean()
                             if area_losses
                             else torch.tensor(0.0, device=self.device))

                # Normalise by number of batches → equivalent to averaging over
                # all images before the patch update
                batch_loss = (
                    max_obj_loss * loss_hypers['lambda_1'] +
                    bbox_loss    * loss_hypers['lambda_2a']
                ) / num_batches

                self.model.zero_grad()
                batch_loss.backward()          # accumulates into adv_patch.grad
                epoch_loss += batch_loss.item()

            # ── Single patch update after full epoch ──────────────────────
            with torch.no_grad():
                adv_patch -= fgsm_epsilon * adv_patch.grad.sign()
                adv_patch.grad.zero_()

                # PGD projection
                norm = torch.norm(adv_patch)
                if norm > pgd_epsilon:
                    adv_patch.mul_(pgd_epsilon / norm)
                adv_patch.clamp_(0.0, 1.0)

            # ── Logging & checkpointing ───────────────────────────────────
            if curr_iter % 10 == 0:
                print(f"  Iter {curr_iter:4d} | Loss: {epoch_loss:.4f} "
                      f"| Last-batch boxes: {last_num_boxes}")

            patch_cpu = adv_patch.detach().cpu()
            torch.save(patch_cpu, iter_patch_dir / f'patch_iter_{curr_iter:04d}.pt')
            if curr_iter % 10 == 0:
                transforms.ToPILImage()(patch_cpu).save(
                    iter_patch_dir / f'patch_iter_{curr_iter:04d}.png'
                )

        # ── Final save ────────────────────────────────────────────────────
        final_patch = adv_patch.detach().cpu()
        torch.save(final_patch, save_path / 'final_patch.pt')
        transforms.ToPILImage()(final_patch).save(save_path / 'final_patch.png')
        print(f"[Saved] Final patch → {save_path / 'final_patch.pt'}")

        del adv_patch
        torch.cuda.empty_cache()
        gc.collect()
        return final_patch


# ---------------------------------------------------------------------------
# Attack Runner
# ---------------------------------------------------------------------------

def run_attack(max_iter, fgsm_epsilon, img_paths, folder_name,
               mini_batch_size=8, warm_start_patch=None):
    """
    Builds a lazy DataLoader from file paths (no pre-loading into RAM),
    then runs the adversarial patch optimisation.
    Returns the final patch tensor for warm-starting the next class.
    """
    loss_hypers = {
        'lambda_1' : 1,   # weight for objectness loss
        'lambda_2a': 5,   # weight for bbox area/spread loss
    }

    dataset    = ImageFolderDataset(img_paths)
    dataloader = DataLoader(
        dataset,
        batch_size  = mini_batch_size,
        shuffle     = True,
        drop_last   = False,
        num_workers = 4,      # set to 0 on Windows if you hit multiprocessing issues
        pin_memory  = True,
    )

    generator = PhantomGenerator(
        model_name     = 'yolov5',
        model_ckpt_path= 'yolov5s.pt',
        max_iter       = max_iter,
    )

    final_patch = generator.generate_phantom(
        dataloader      = dataloader,
        loss_hypers     = loss_hypers,
        pgd_epsilon     = 70.0,
        fgsm_epsilon    = fgsm_epsilon,
        folder_name     = folder_name,
        warm_start_patch= warm_start_patch,
    )

    del generator, dataloader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return final_patch


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ── Configuration ─────────────────────────────────────────────────────
    MINI_BATCH_SIZE = 8       # images per mini-batch  (tune to fit VRAM)
    ITERATIONS      = 1000    # patch update steps per class
    EPSILON         = 0.005   # FGSM step size

    GLOBAL_PATCH_PATH = Path('global_patch.pt')

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
    skip_until_class = 'refrigerator'
    found_skip       = False

    global_patch = None
    if GLOBAL_PATCH_PATH.exists():
        global_patch = torch.load(GLOBAL_PATCH_PATH)
        print(f"[Resume] Loaded global patch from {GLOBAL_PATCH_PATH}")
    else:
        print("[Start] No global patch found — cold start.")

    # ── Main loop ─────────────────────────────────────────────────────────
    for root_dir in image_directories:
        class_folders = sorted(glob.glob(os.path.join(root_dir, '*/')))

        for cls_path in class_folders:
            folder_name = Path(cls_path).name

            # Skip until the target class is reached
            if not found_skip:
                if folder_name == skip_until_class:
                    print(f"[Skip] Resuming from class: {folder_name}")
                    found_skip = True
                else:
                    continue

            print(f"\n[Processing] Class: {folder_name}")
            is_important = folder_name in classes_important

            # Collect image paths — NO loading into RAM here
            img_paths = sorted(glob.glob(os.path.join(cls_path, '*.jpg')))

            if not img_paths:
                print(f"  [Skip] No .jpg images found.")
                continue

            print(f"  Found {len(img_paths)} images | Important: {is_important}")

            # Run attack — DataLoader handles lazy loading & mini-batching
            global_patch = run_attack(
                max_iter        = ITERATIONS,
                fgsm_epsilon    = EPSILON,
                img_paths       = img_paths,
                folder_name     = folder_name,
                mini_batch_size = MINI_BATCH_SIZE,
                warm_start_patch= global_patch,
            )

            # Persist updated patch for crash recovery
            torch.save(global_patch, GLOBAL_PATCH_PATH)
            print(f"[Checkpoint] Global patch saved → {GLOBAL_PATCH_PATH}")

    print("\n[Done] Pipeline execution complete.")