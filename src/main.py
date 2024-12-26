import time
import torch
import numpy as np
from dataset import ImageDataset
from model import initialize_biomedclip
from retrieval import FaissSearcher
from evaluate import evaluate_search
from torch.utils.data import DataLoader
import argparse
from cross_attention_encoder import CrossAttentionEncoderLayer, CrossAttentionEncoder, initialize_cross_attention_encoder
from self_attention_encoder import SelfAttentionEncoderLayer, SelfAttentionEncoder, initialize_image_self_attention_encoder, initialize_text_self_attention_encoder
from torch.nn import LayerNorm

def main(calc_embeddings, CA_query, cross_attention_encoder_checkpoint_path):
    print(f"Running experiment")

    # データセットのロード
    query_dataset = ImageDataset(f'../data/query.csv')
    query_loader = DataLoader(query_dataset, batch_size=64, shuffle=False)
    candidate_dataset = ImageDataset(f'../data/candidate.csv')
    candidate_loader = DataLoader(candidate_dataset, batch_size=64, shuffle=False)
    
    # biomedclipモデルの定義
    model = initialize_biomedclip().to(device)
    model.eval()

    # cross-attentionモデルの定義
    cross_attn_encoder = initialize_cross_attention_encoder(cross_attention_encoder_checkpoint_path)
    
    # 検索エンジンの初期化
    searcher = FaissSearcher(model, cross_attn_encoder, device, calc_embeddings, CA_query)

    # 検索を実行・評価
    start_time = time.time()
    results = searcher.perform_search(query_loader, candidate_loader)
    evaluation = evaluate_search(results, query_loader, candidate_loader)
    end_time = time.time()
    print(f"評価にかかった時間: {end_time - start_time}秒")

    return evaluation

if __name__ == "__main__":
    # argparseでコマンドライン引数を定義
    parser = argparse.ArgumentParser(description="cbmir")
    parser.add_argument("--calc_embeddings", type=bool, default=False,
                        choices=[True, False])
    parser.add_argument("--CA_query", type=str, default="image",
                        choices=["text", "image"])
    parser.add_argument("--cross_attention_encoder_checkpoint_path", type=str, default="../results/checkpoints/cross_attention_encoder/epoch_12.pt"),

    # 引数をパース
    args = parser.parse_args()

    # デバイスの設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # 評価
    main(
        calc_embeddings=args.calc_embeddings,
        CA_query=args.CA_query,
        cross_attention_encoder_checkpoint_path=args.cross_attention_encoder_checkpoint_path
    )