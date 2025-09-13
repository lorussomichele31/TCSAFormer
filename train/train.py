
import torch

from presets import build_cfg
from train_utils import set_seed, train_one_dataset


def main():
    PRESET = "cvc_clinicdb"   # "isic2018" | "cvc_clinicdb" | "synapse"

    cfg, seed = build_cfg(PRESET)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_one_dataset(cfg, device)

if __name__ == "__main__":
    main()
