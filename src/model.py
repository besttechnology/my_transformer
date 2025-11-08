import torch
import torch.nn as nn
import math
from src.transformer_components import (
    PositionalEncoding,
    MultiHeadAttention,
    PositionWiseFeedForward,
    SublayerConnection,
    EncoderLayer
)

# ---------------------------------------------------------------
# 模块五：Decoder Layer
# ---------------------------------------------------------------
# 这是 80-90 分的关键。
# DecoderLayer 比 EncoderLayer 多一个 "Encoder-Decoder Attention"

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # 1. 带掩码的 MHA (Masked Multi-Head Attention)
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        
        # 2. Encoder-Decoder 注意力
        # Q 来自 Decoder (上一层)，K 和 V 来自 Encoder 的输出
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        
        # 3. FFN
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x: 目标序列 (decoder input)
        memory: 编码器的最终输出 (K 和 V)
        src_mask: 源序列的 padding mask
        tgt_mask: 目标序列的 "subsequent" mask + padding mask
        """
        # 1. 应用带掩码的 MHA (Q, K, V 都是 x)
        x = self.sublayer1(x, lambda x: self.masked_self_attn(x, x, x, tgt_mask))
        
        # 2. 应用 Encoder-Decoder 注意力
        # Q 是 x, K 和 V 是 memory
        x = self.sublayer2(x, lambda x: self.enc_dec_attn(x, memory, memory, src_mask))
        
        # 3. FFN
        x = self.sublayer3(x, self.feed_forward)
        
        return x

# ---------------------------------------------------------------
# 模块六：Encoder (封装)
# ---------------------------------------------------------------
# 将 N 个 EncoderLayer 堆叠起来

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model) # 最后的归一化
        
    def forward(self, x, mask):
        # 1. Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        # 2. 逐层传递
        for layer in self.layers:
            x = layer(x, mask)
            
        # 3. 最终 Norm
        return self.norm(x)

# ---------------------------------------------------------------
# 模块七：Decoder (封装)
# ---------------------------------------------------------------
# 将 N 个 DecoderLayer 堆叠起来

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model) # 最后的归一化
        
    def forward(self, x, memory, src_mask, tgt_mask):
        # 1. Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        # 2. 逐层传递
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            
        # 3. 最终 Norm
        return self.norm(x)

# ---------------------------------------------------------------
# 模块八：完整的 Transformer (Seq2Seq)
# ---------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        
        # 最终的线性层，将 Decoder 的输出映射回目标词表大小
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数 (可选，但推荐)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src: 源序列 (batch_size, src_seq_len)
        tgt: 目标序列 (batch_size, tgt_seq_len)
        src_mask: 源 padding mask
        tgt_mask: 目标 padding + subsequent mask
        """
        # 1. Encoder 编码源序列
        memory = self.encoder(src, src_mask)
        
        # 2. Decoder 解码
        decoder_output = self.decoder(tgt, memory, src_mask, tgt_mask)
        
        # 3. 最终输出
        output = self.final_linear(decoder_output)
        
        return output

    # --- 掩码 (Masking) ---
    # 这是 Transformer 非常关键的部分
    
    def create_padding_mask(self, seq, pad_idx, device):
    # seq 形状: (batch_size, seq_len)
    # 返回 (batch_size, 1, 1, seq_len) 以便广播到 (batch_size, num_heads, seq_len, seq_len)
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def create_subsequent_mask(self, seq_len, device):
        # 创建一个上三角矩阵
        # 形状: (1, 1, seq_len, seq_len)
        mask = (torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) == 0)
        return mask.unsqueeze(0).unsqueeze(0) # 增加 batch 和 head 维度
        
    def create_masks(self, src, tgt, pad_idx, device):
        # 1. 源 padding mask
        src_mask = self.create_padding_mask(src, pad_idx, device)
        
        # 2. 目标 padding mask
        tgt_pad_mask = self.create_padding_mask(tgt, pad_idx, device)
        
        # 3. 目标 subsequent mask (防止 "看到未来")
        tgt_seq_len = tgt.size(1)
        tgt_sub_mask = self.create_subsequent_mask(tgt_seq_len, device)
        
        # 4. 合并目标掩码
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return src_mask, tgt_mask

if __name__ == "__main__":
    # 快速测试我们的模型
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 超参数 (来自作业 PDF)
    SRC_VOCAB = 1000
    TGT_VOCAB = 1200
    D_MODEL = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    D_FF = 512
    
    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF
    ).to(DEVICE)
    
    # 假设的输入
    src_seq = torch.randint(1, SRC_VOCAB, (8, 10)).to(DEVICE) # (batch_size=8, seq_len=10)
    tgt_seq = torch.randint(1, TGT_VOCAB, (8, 12)).to(DEVICE) # (batch_size=8, seq_len=12)
    
    # 假设的 padding 索引
    PAD_IDX = 0
    src_seq[0, 5:] = PAD_IDX
    tgt_seq[0, 6:] = PAD_IDX
    
    # 创建掩码
    src_mask, tgt_mask = model.create_masks(src_seq, tgt_seq, PAD_IDX, DEVICE)
    
    print(f"源序列形状: {src_seq.shape}")
    print(f"目标序列形状: {tgt_seq.shape}")
    print(f"源掩码形状: {src_mask.shape}")
    print(f"目标掩码形状: {tgt_mask.shape}")
    
    # 前向传播
    output = model(src_seq, tgt_seq, src_mask, tgt_mask)
    
    print(f"模型输出形状: {output.shape}") # (batch_size, tgt_seq_len, tgt_vocab_size)
    assert output.shape == (8, 12, TGT_VOCAB)
    print("\n模型搭建测试通过！")