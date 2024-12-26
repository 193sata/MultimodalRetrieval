import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import LayerNorm, Dropout, Linear
from torch.nn.modules.transformer import MultiheadAttention
from typing import Optional, Callable, Union

class CrossAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True
    ):
        super(CrossAttentionEncoderLayer, self).__init__()
        self.cross_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias
        )
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=bias)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if isinstance(activation, str):
            if activation == "relu":
                self.activation = F.relu
            elif activation == "gelu":
                self.activation = F.gelu
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation
        
    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Cross-Attention (query: text, key & value: image)
        if self.norm_first:
            query = query + self._cross_attn_block(
                self.norm1(query), key_value, src_mask, src_key_padding_mask
            )
            query = query + self._ff_block(self.norm2(query))
        else:
            query = self.norm1(
                query + self._cross_attn_block(query, key_value, src_mask, src_key_padding_mask)
            )
            query = self.norm2(query + self._ff_block(query))
        return query

    def _cross_attn_block(
        self, query: Tensor, key_value: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        attn_output, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout1(attn_output)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
        use_position_embedding: bool = False,  # 位置埋め込みを使用するかのフラグ
        max_img_tokens: int = 197,             # 画像トークンの数
        max_text_tokens: int = 256             # テキストトークンの数
    ):
        super(CrossAttentionEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        self.use_position_embedding = use_position_embedding
        # 画像とテキストそれぞれに異なる位置埋め込みを設定
        if self.use_position_embedding:
            self.img_position_embedding = nn.Parameter(torch.randn(1, max_img_tokens, encoder_layer.cross_attn.embed_dim))
            self.text_position_embedding = nn.Parameter(torch.randn(1, max_text_tokens, encoder_layer.cross_attn.embed_dim))

    def forward(
        self,
        query: Tensor,  # Text embeddings (as query input)
        key_value: Tensor,  # Image embeddings (as key and value input)
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = query

        # 各層のCross-Attentionに位置埋め込みを適用
        if self.use_position_embedding:
            output = output + self.img_position_embedding[:, :output.size(1), :]
            key_value = key_value + self.text_position_embedding[:, :key_value.size(1), :]

        for layer in self.layers:
            output = layer(output, key_value, mask, key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)
        
        # CLSトークン（通常は最初のトークン）を抜き出して、(batch_size, 1, d_model) の形状にする
        cls_output = output[:, 0, :].unsqueeze(1)  # (batch_size, 1, d_model)
        
        return cls_output

def initialize_cross_attention_encoder(checkpoint_path):
    d_model = 768
    nhead = 8
    dim_feedforward = 2048
    num_layers = 6
    dropout = 0.1
    # Define a single cross-attention encoder layer
    cross_attn_layer = CrossAttentionEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="relu",
        batch_first=True
    )
    # Define the full cross-attention encoder with multiple layers
    cross_attn_encoder = CrossAttentionEncoder(
        encoder_layer=cross_attn_layer,
        num_layers=num_layers,
        norm=LayerNorm(d_model), 
        use_position_embedding=False # 位置埋め込み
    )
    cross_attn_encoder.load_state_dict(torch.load(checkpoint_path))
    cross_attn_encoder.eval()
    
    return cross_attn_encoder