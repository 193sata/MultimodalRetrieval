import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import ImageDataset
from model import initialize_biomedclip
import argparse

def create_embeddings(csv_path, data_type, batch_size=64, device='cuda:0'):
    """
    指定されたデータセットからBioMedCLIPの画像またはテキストの埋め込みを計算し、.npy形式で保存します。

    Args:
        csv_path (str): csvファイルのパス
        data_type (str): 埋め込みの種類（"image" or "text"）
        batch_size (int): バッチサイズ（デフォルト: 64）
        device (str): 使用するデバイス（"cuda:0" or "cpu"）
    """

    if data_type in ["image", "image_patch"]:
        _data_type = "image"
    else:
        _data_type = "text"

    # 中間層の出力を保存するためのリスト
    # hidden_states = []
    # フォワードフックを定義
    def hook(module, input, output):
        hidden_states.append(output)

    # モデルのロード
    model = get_model().to(device)
    model.eval()
    if data_type == "image_patch":
        model.visual.trunk.blocks[-1].register_forward_hook(hook) # パッチごと

    # データセットとデータローダーの初期化
    dataset = ImageDataset(csv_path, _data_type)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []

    # 埋め込みの計算
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            hidden_states = []
            if data_type == "image":
                features = model.encode_image(inputs).cpu().numpy() # 通常
            elif data_type == "image_patch":
                model(inputs)# パッチごと
                features = hidden_states[-1].cpu().detach().numpy() #パッチごとの埋め込み
            elif data_type == "text":
                features = model.encode_text(inputs).cpu().numpy() # 通常
            elif data_type == "text_token":
                features = model.text.transformer(inputs, output_hidden_states=False).last_hidden_state.cpu().detach().numpy() # トークンごと
            print(features.shape, flush=True)
            embeddings.append(features)

    # 埋め込みを連結して保存
    embeddings = np.vstack(embeddings)
    output_file = f'../data/embeddings/{data_type}_{csv_path[8:-4]}_embeddings.npy'
    np.save(output_file, embeddings)
    print(f"埋め込みが{output_file}に保存されました。")

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="BioMedCLIPによる埋め込み計算と保存")
    parser.add_argument("--csv_path", type=str, default="../data/train.csv", help="csvファイルのパス")
    parser.add_argument("--data_type", type=str, default="image", choices=["image", "text", "image_patch", "text_token"], help="取得する埋め込みの種類")
    parser.add_argument("--batch_size", type=int, default=64, help="バッチサイズ")
    parser.add_argument("--device", type=str, default="cuda:0", help="使用するデバイス")
    args = parser.parse_args()

    # 埋め込みの保存プロセスを実行
    create_embeddings(
        csv_path=args.csv_path,
        data_type=args.data_type,
        batch_size=args.batch_size,
        device=args.device
    )