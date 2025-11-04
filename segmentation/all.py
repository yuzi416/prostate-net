import os
import cv2
import numpy as np
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine.model import revert_sync_batchnorm

def process_image(model, img_path, out_file=None, opacity=0.5, with_labels=False,
                  title=None, crop_out_file=None, padding=32, crop_size=(256,256)):
    result = inference_model(model, img_path)
    show_result_pyplot(model, img_path, result,
                       title=title, opacity=opacity,
                       with_labels=with_labels, draw_gt=False,
                       show=(out_file is None), out_file=out_file)

    if crop_out_file:
        mask = result.pred_sem_seg.data[0].cpu().numpy()
        mask_bin = (mask == 1).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            img = cv2.imread(img_path)
            h_img, w_img = img.shape[:2]
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(w_img, x + w + padding)
            y1 = min(h_img, y + h + padding)
            cropped = img[y0:y1, x0:x1]
            resized = cv2.resize(cropped, crop_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(crop_out_file, resized)

def pro_crop(img_dir, config_file, checkpoint_file, device='cuda:0',
                      out_dir=None, crop_out_dir=None, opacity=0.5, with_labels=False, title=None):
    model = init_model(config_file, checkpoint_file, device=device)
    if device.startswith('cpu'):
        model = revert_sync_batchnorm(model)

    os.makedirs(out_dir, exist_ok=True) if out_dir else None
    os.makedirs(crop_out_dir, exist_ok=True) if crop_out_dir else None

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        out_file = os.path.join(out_dir, f"seg_{fname}") if out_dir else None
        crop_file = os.path.join(crop_out_dir, fname) if crop_out_dir else None
        process_image(model, img_path, out_file=out_file, opacity=opacity,
                      with_labels=with_labels, title=title, crop_out_file=crop_file)

    return crop_out_dir
