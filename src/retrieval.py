import faiss
import torch
import numpy as np

class FaissSearcher:
    def __init__(self, model, cross_attn_encoder, device,  calc_embeddings=False, CA_query="text"):
        """
        model: 特徴ベクトルを抽出するためのPyTorchモデル
        device: 使用するデバイス（CPU or GPU）
        distance_type: "L2" or "cosine" を指定して距離のタイプを選択
        """
        self.model = model
        self.device = device
        self.index = None
        self.calc_embeddings = calc_embeddings
        self.cross_attn_encoder = cross_attn_encoder
        self.CA_query = CA_query
        self.model.visual.trunk.blocks[-1].register_forward_hook(self.hook) #最終ブロックにフックを登録

    # フォワードフックを定義
    def hook(self, module, input, output):
        self.hidden_states.append(output)
    
    def _extract_image_features(self, features, inputs):
        # 中間層の出力を保存するためのリスト
        self.hidden_states = []
        self.model(inputs)
        if not isinstance(features, list):
            features = list(features)  # featuresがリストでない場合リスト化
        features.append(self.hidden_states[-1].cpu().numpy()) #パッチごとの埋め込み
        return features

    def _extract_text_features(self, features, inputs):
        if not isinstance(features, list):
            features = list(features)  # featuresがリストでない場合リスト化
        features.append(self.model.text.transformer(inputs, output_hidden_states=False).last_hidden_state.cpu().detach().numpy())
        return features

    def _extract_features(self, loader):
        with torch.no_grad():
            # 埋め込みの生成とL2正規化
            image_features = []
            text_features = []
            for image_inputs, text_inputs, _ in loader:
                image_inputs = image_inputs.to(self.device)
                text_inputs = text_inputs.to(self.device)
                image_features = self._extract_image_features(image_features, image_inputs)
                text_features = self._extract_text_features(text_features, text_inputs)
            # image_features と text_features をそれぞれ縦に結合して NumPy 配列に変換
            image_features = np.concatenate([np.array(feat).reshape(-1) if np.array(feat).ndim == 1 else np.array(feat) for feat in image_features], axis=0)
            text_features = np.concatenate([np.array(feat).reshape(-1) if np.array(feat).ndim == 1 else np.array(feat) for feat in text_features], axis=0)

            # print(f'image_features shape: {image_features.shape}')  # (x, 197, 768)
            # print(f'text_features shape: {text_features.shape}')    # (x, 256, 768)

            # それをcross-attnに入力してfeaturesを生成
            if self.CA_query == "text":
                features = self.cross_attn_encoder(torch.tensor(text_features), torch.tensor(image_features)).squeeze(1).detach().numpy()                  
            elif self.CA_query == "image":
                features = self.cross_attn_encoder(torch.tensor(image_features), torch.tensor(text_features)).squeeze(1).detach().numpy()                  
        return np.vstack(features)

    def _normalize_features(self, features):
        # Cos類似度のために特徴ベクトルをL2正規化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / norms

    def _build_index(self, candidate_features):
        dim = candidate_features.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Cos類似度は内積を使う
        self.index.add(candidate_features)
        # print(f"index.ntotal: {self.index.ntotal}")

    def perform_search(self, query_loader, candidate_loader):
        # 検索候補の特徴量を抽出
        if self.calc_embeddings:
            candidate_features = self._extract_features(candidate_loader)
            # concatenateの場合の処理は_extract_features内で行う
        else:
            # パッチごと、トークンごとの埋め込みをロード
            candidate_image_features = np.load('../data/embeddings/image_patch_candidate_embeddings.npy')
            candidate_text_features = np.load('../data/embeddings/text_token_candidate_embeddings.npy')

            #cross-attnに入力し、candidate_featuresを獲得
            if self.CA_query == "text":
                candidate_features = self.cross_attn_encoder(torch.tensor(candidate_text_features), torch.tensor(candidate_image_features)).squeeze(1).detach().numpy()
            elif self.CA_query == "image":
                candidate_features = self.cross_attn_encoder(torch.tensor(candidate_image_features), torch.tensor(candidate_text_features)).squeeze(1).detach().numpy()

        # Cos類似度を使う場合は特徴ベクトルを正規化
        candidate_features = self._normalize_features(candidate_features)

        # インデックスを構築
        self._build_index(candidate_features)
        print(f'self.index.ntotal: {self.index.ntotal}')

        # クエリの特徴量を抽出
        query_features = self._extract_features(query_loader)
        print(query_features.shape)

        # Cos類似度の場合はクエリの特徴ベクトルも正規化
        query_features = self._normalize_features(query_features)

        # FAISSによる検索
        np.set_printoptions(threshold=10000) # npのprint上限を大きく
        D, I = self.index.search(query_features, 100)  # 上位100件までの結果を返す
        # print(f"検索結果のインデックス：\n{I}") # [50,100]
        # print(len(I)) # 50
        # print(len(I[0])) # 100

        return I