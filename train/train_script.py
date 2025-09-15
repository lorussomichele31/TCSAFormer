import torch

from presets import build_cfg
from train_utils import set_seed, train_one_dataset


def main():
    PRESET = "cvc_clinicdb-finetune"   # "isic2018" | "cvc_clinicdb" | "synapse"

    cfg = build_cfg(PRESET)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_one_dataset(cfg, device)

if __name__ == "__main__":
    main()
