import os
import fileinput
import glob
import pixellib
from pixellib.semantic import semantic_segmentation
import segment
import cv2
import numpy as np
import sys
import calendar
import time
import logging  # Added for log file support

# --- Logging Configuration ---
# This sets up the format to include a timestamp for better traceability
log_filename = "segmentation_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

# --- Keras Import Patching ---
FILE_PATH = "Repo\\DetStorm\\.venv\\Lib\\site-packages\\pixellib\\semantic\\deeplab.py"


def patch_file(path, old, new):
    for line in fileinput.input(path, inplace=True):
        print(line.replace(old, new), end='')


patch_file(FILE_PATH, "tensorflow.python.keras", "tensorflow.keras")
patch_file(FILE_PATH, "tensorflow.keras.utils.layer_utils import get_source_inputs",
           "tensorflow.python.keras.utils.layer_utils import get_source_inputs")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Initialization ---
segment_image = semantic_segmentation()
segment_image.load_ade20k_model("downloaded_models/deeplabv3_xception65_ade20k.h5")


img_directory = '.\\bdd100k_images_100k\\100k\\test'
out_directory = '.\\out_segments'
current_out_dir = os.path.join(out_directory, str(calendar.timegm(time.gmtime())))

if not os.path.exists(current_out_dir):
    os.makedirs(current_out_dir)

logging.info(f"Output directory initialized: {current_out_dir}")

# --- Processing Loop ---
idx_counter = 0

for impath in glob.iglob(os.path.join(img_directory, '*')):
    return_masks, return_ims = segment.segmentation_mask(segment_image, impath)
    idx_counter += 1

    # Calculate mask count for the current image
    current_image_mask_count = sum(len(masks) for masks in return_masks.values())
    img_name = os.path.splitext(os.path.basename(impath))[0]

    # Requirement 2: Log masks present for each image
    logging.info(f"Image {idx_counter}: {img_name} | Masks count: {current_image_mask_count}")

    for cls, masks in return_masks.items():
        cls_folder = os.path.join(current_out_dir, cls)

        if len(masks) > 0 and not os.path.exists(cls_folder):
            os.makedirs(cls_folder)

        for i in range(len(masks)):
            file_base = f"{img_name}_{i}"
            cv2.imwrite(os.path.join(cls_folder, f"{file_base}.jpg"), return_ims[cls][i])
            np.save(os.path.join(cls_folder, f"{file_base}.npy"), masks[i])

# Requirement 1: Log total number of images
logging.info("-" * 40)
logging.info("Processing Summary")
logging.info(f"Total number of images processed: {idx_counter}")
logging.info("-" * 40)