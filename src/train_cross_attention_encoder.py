import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from cross_attention_encoder import CrossAttentionEncoderLayer, CrossAttentionEncoder
from loss import CustomSupConLoss
from dataset import TrainingDataset
from torch.nn import LayerNorm
import os

# 保存先ディレクトリ
save_dir = "../results/checkpoints/cross_attention_encoder"
os.makedirs(save_dir, exist_ok=True)

# パラメータ設定
batch_size = 32
num_epochs = 30
learning_rate = 1e-4
temperature = 0.07

# データセットとデータローダーの準備
train_dataset = TrainingDataset(
    csv_path='../data/train.csv',
    image_embeddings_path='../data/embeddings/image_patch_train_embeddings.npy',
    text_embeddings_path='../data/embeddings/text_token_train_embeddings.npy'
)
val_dataset = TrainingDataset(
    csv_path='../data/validate.csv',
    image_embeddings_path='../data/embeddings/image_patch_validate_embeddings.npy',
    text_embeddings_path='../data/embeddings/text_token_validate_embeddings.npy'
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
torch.manual_seed(42)  # シード固定
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) # 検証データもシャッフルするが、固定されたシードで再現性を確保
torch.manual_seed(torch.initial_seed()) # ここでシードをリセットしておくと、以降の部分での影響を防げる

# クロスアテンションエンコーダの構築
d_model = 768
nhead = 8
dim_feedforward = 2048
num_layers = 6
dropout = 0.1

cross_attn_layer = CrossAttentionEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation="relu",
    batch_first=True
)
model = CrossAttentionEncoder(
    encoder_layer=cross_attn_layer,
    num_layers=num_layers,
    norm=LayerNorm(d_model), 
    use_position_embedding=False # 位置埋め込み
)

# デバイスの設定
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 損失関数とオプティマイザの準備
criterion = CustomSupConLoss(temperature=temperature).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 検証損失の最小値を追跡
best_val_loss = float('inf')

# 学習と検証ループ
for epoch in range(num_epochs):
    # ======= 学習フェーズ =======
    model.train()
    total_train_loss = 0

    for image_embeddings, text_embeddings, labels in train_loader:
        # デバイスに転送
        image_embeddings = image_embeddings.to(device)
        text_embeddings = text_embeddings.to(device)
        labels = labels.to(device)

        # クロスアテンションエンコーダで画像とテキストの統合埋め込みを計算
        combined_embeddings = model(query=image_embeddings, key_value=text_embeddings)
        combined_embeddings = combined_embeddings.squeeze(1)  # (batch_size, 768)

        # 損失計算
        loss = criterion(combined_embeddings, labels)

        # 勾配のリセットとバックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ======= 検証フェーズ =======
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for image_embeddings, text_embeddings, labels in val_loader:
            # デバイスに転送
            image_embeddings = image_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)
            labels = labels.to(device)

            # クロスアテンションエンコーダで画像とテキストの統合埋め込みを計算
            combined_embeddings = model(query=image_embeddings, key_value=text_embeddings)
            combined_embeddings = combined_embeddings.squeeze(1)  # (batch_size, 768)

            # 損失計算
            val_loss = criterion(combined_embeddings, labels)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    # エポックごとの損失の表示
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}", flush=True)
    
    # 最小検証損失の更新とモデル保存
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"検証損失が更新されました。モデルの重みを {model_path} に保存しました", flush=True)