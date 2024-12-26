import pandas as pd
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from medclip import MedCLIPProcessor
from open_clip import get_tokenizer
import torch
from open_clip import create_model_and_transforms
import json
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from utils import extract_sections, pad_tensors
import numpy as np
import yaml
import os

class ImageDataset(Dataset):
    def __init__(self, csv_path, config_path="config.yaml"):
        # 設定ファイルを読み込み
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.image_base = config["paths"]["image_base"]
        self.text_base = config["paths"]["text_base"]
        self.data = pd.read_csv(csv_path)

        model_name = "biomedclip_local"
        with open("../pretrained/biomedclip/checkpoints/open_clip_config.json", "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]
        if (not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None):
            _MODEL_CONFIGS[model_name] = model_cfg
        _, _, self.preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained="../pretrained/biomedclip/checkpoints/open_clip_pytorch_model.bin",
            **{f"image_{k}": v for k, v in preprocess_cfg.items()}
        )
        self.tokenizer = get_tokenizer(model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_inputs, label = self.return_images(row)
        text_inputs, _ = self.return_texts(row)
        return image_inputs, text_inputs, label
            
    def return_images(self, row):
        # image_path = f"/user/arch/sata/CBIR/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg"
        image_path = os.path.join(
            self.image_base,
            f"p{str(row['subject_id'])[:2]}",
            f"p{row['subject_id']}",
            f"s{row['study_id']}",
            f"{row['dicom_id']}.jpg"
        )
        label = row.iloc[3:].astype(float).values
        #読み込み
        image = Image.open(image_path).convert('RGB')
        #前処理
        return torch.stack([self.preprocess(image)]).squeeze(), label

    def return_texts(self, row):
        # txt_path = f"/home1/user/sata/MIMIC-CXR/physionet.org/files/mimic-cxr/2.0.0/files/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}.txt"
        txt_path = os.path.join(
            self.text_base,
            f"p{str(row['subject_id'])[:2]}",
            f"p{row['subject_id']}",
            f"s{row['study_id']}.txt"
        )
        label = row.iloc[3:].astype(float).values
        #読み込み
        with open(txt_path, 'r') as txt_file:
            text = txt_file.read()
            text = extract_sections(text) # Findings & Impressionセクション

        return self.tokenizer(text, context_length=256).squeeze(), label

class TrainingDataset(Dataset):
    def __init__(self, csv_path, image_embeddings_path, text_embeddings_path):
        # CSVファイルを読み込み
        self.data = pd.read_csv(csv_path)

        # 画像とテキストの埋め込みを読み込み
        self.image_embeddings = np.load(image_embeddings_path)  # (1245, 197, 768)
        self.text_embeddings = np.load(text_embeddings_path)  # (1245, 256, 768)

        # ラベルを取得
        self.labels = self.data.iloc[:, 3:].values  # ラベル情報 (1245, 5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # インデックスに基づき画像・テキスト埋め込みとラベルを取得
        image_embedding = torch.tensor(self.image_embeddings[idx], dtype=torch.float32)  # (197, 768)
        text_embedding = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)  # (256, 768)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # (5,)

        return image_embedding, text_embedding, label