import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CustomSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: 特徴ベクトルのテンソルで、形状は [batch_size, dim]
            labels: ワンホット形式のクラスラベルで、形状は [batch_size, num_classes]
        Returns:
            loss: スカラーの損失値
        """
        features = F.normalize(features, p=2, dim=1)  # 正規化の追加
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        batch_size = features.shape[0]
        
        # ラベルからマスクを生成（同じクラスのサンプルは1、異なるクラスのサンプルは0）
        mask = torch.matmul(labels, labels.T).float().to(device)

        # 特徴ベクトル間の類似度行列（内積）を計算し、温度でスケーリング
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # 数値的な安定性のために各行の最大値を引く
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 自己相似を防ぐためのマスク作成
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask  # 対角成分を0にする

        # ソフトマックス計算のための準備
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 正例に対する平均の計算
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # エッジケース対策
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # 最終的な損失の計算
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss