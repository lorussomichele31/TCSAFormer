from train.presets import build_cfg
from inference_utils import load_model_for_inference, infer_image, infer_folder

cfg = build_cfg( "cvc_clinicdb-finetune")  # or your finetune preset (resolution matched)
model = load_model_for_inference(cfg, "checkpoints/cvc_clinicdb_ft320x10/cvc_clinicdb_best.pt")

# Single image
"""
mask = infer_image(cfg, model,
                   image_path="train/datasets/CVC-ClinicDB/images/test/552.png",
                   out_mask_path="inference/masks/552.png",
                   out_overlay_path="inference/overlays/552.png",
                   threshold=0.5, keep_size=True)
"""
# Folder

infer_folder(cfg,
             ckpt_path="checkpoints/cvc_clinicdb/cvc_clinicdb_best.pt",
             in_dir="train/datasets/CVC-ClinicDB/images/test",
             out_dir_masks="inference/masks",
             out_dir_overlays="inference/overlays",
             threshold=0.5, keep_size=True)
