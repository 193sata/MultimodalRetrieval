import torch
import torchvision.models as models
import random
import numpy as np
from utils import set_random_seed
import json
from open_clip import create_model_and_transforms
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

def initialize_biomedclip(seed=None):
    if seed is not None:
        # シードが指定されている場合はシードを設定
        set_random_seed(seed)

    model_name = "biomedclip_local"
    with open("../pretrained/biomedclip/checkpoints/open_clip_config.json", "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None):
        _MODEL_CONFIGS[model_name] = model_cfg

    model, _, _ = create_model_and_transforms(
        model_name=model_name,
        pretrained="../pretrained/biomedclip/checkpoints/open_clip_pytorch_model.bin",
        **{f"image_{k}": v for k, v in preprocess_cfg.items()}
    )
    # 全結合層を無効化
    model.fc = torch.nn.Identity()
    return model

