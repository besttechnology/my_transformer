import torch
import torch.nn as nn
import math
import numpy as np

# ---------------------------------------------------------------
# 模块一：位置编码 (Positional Encoding) [cite: 10, 55]
# ---------------------------------------------------------------
# 作业要求实现位置编码 。
# 我们使用 "Attention Is All You Need" [cite: 107] 中的正弦/余弦公式 [cite: 55]。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_prob, max_len=5000):
        """
        参数:
        d_model: 模型的维度 (embedding dimension)
        max_len: 预先计算的最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout_prob)

        # 创建一个 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # (d_model/2)
        # 计算 1 / (10000^(2i / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数索引使用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数索引使用 cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加一个 batch 维度 (1, max_len, d_model)
        # register_buffer 意味着这个张量是模型的状态，但不被视为模型参数（即不会被优化器更新）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x: 输入张量，形状为 (batch_size, seq_len, d_model)
        """
        # 将位置编码加到输入张量上
        # x.size(1) 是 seq_len，我们只取 pe 中前 seq_len 个位置
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ---------------------------------------------------------------
# 模块二：多头自注意力 (Multi-Head Self-Attention) [cite: 10, 47]
# ---------------------------------------------------------------
# 这是 Transformer 的核心。
# 它首先包含一个 Scaled Dot-Product Attention [cite: 44]

class ScaledDotProductAttention(nn.Module):
    """ 
    作业中提到的 Scaled Dot-Product Attention [cite: 44] 
    公式: Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) * V [cite: 48]
    """
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        # 1. 计算 QK^T / sqrt(d_k)
        # Q, K, V 形状: (batch_size, num_heads, seq_len, d_k)
        # scores 形状: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # [cite: 48]
        
        # 2. 应用掩码 (Mask)
        if mask is not None:
            # 掩码为 0 的地方设为 -1e9 (一个很小的数)
            scores = scores.masked_fill(mask == 0, -1e9) # [cite: 67, 68]
            
        # 3. Softmax
        attn_weights = torch.softmax(scores, dim=-1) # [cite: 48, 69]
        attn_weights = self.dropout(attn_weights)
        
        # 4. 乘以 V
        # output 形状: (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attn_weights, V) # [cite: 48]
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    作业要求的 Multi-Head Attention [cite: 10, 47]
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model  # 模型的总维度
        self.num_heads = num_heads  # 头的数量 [cite: 49]
        self.d_k = d_model // num_heads # 每个头的维度 (d_k)
        
        # Q, K, V 的线性映射层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
        # 最终的线性输出层
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. 线性映射 Q, K, V
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)
        
        # 2. 拆分成多头 [cite: 49]
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算 Scaled Dot-Product Attention
        # x 形状: (batch_size, num_heads, seq_len, d_k)
        x, attn_weights = self.attention(Q, K, V, mask=mask)
        
        # 4. 合并多头 [cite: 49]
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 最终线性输出
        # (batch_size, seq_len, d_model)
        output = self.W_o(x)
        
        return output, attn_weights

# ---------------------------------------------------------------
# 模块三：Position-wise Feed-Forward Network (FFN) [cite: 10, 50]
# ---------------------------------------------------------------
class PositionWiseFeedForward(nn.Module):
    """
    作业要求的 FFN 。
    它是一个两层的 MLP [cite: 51]
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: 输入输出维度
        d_ff: 隐藏层维度 (通常是 d_model * 4)
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # out: (batch_size, seq_len, d_model)
        return x

# ---------------------------------------------------------------
# 模块四：残差 + LayerNorm (Residual + LayerNorm) [cite: 10, 52]
# ---------------------------------------------------------------
# 这不是一个单独的模块，而是应用在 MHA 和 FFN 之后的 "包装器" [cite: 53]。
# 我们可以创建一个子层连接 (SublayerConnection) 模块来处理这个。
# 我们将使用 PyTorch 内置的 LayerNorm

class SublayerConnection(nn.Module):
    """
    实现 "Add & Norm" [cite: 10, 52]
    """
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model) # [cite: 52]
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        x: 上一层的输入
        sublayer: 要应用残差连接的模块 (例如 MHA 或 FFN)
        """
        # 1. 应用子层 (MHA 或 FFN)
        sublayer_output = sublayer(x)
        
        # 如果 sublayer 是 MHA，它会返回 (output, weights)
        if isinstance(sublayer_output, tuple):
            sublayer_output = sublayer_output[0]
            
        # 2. Add: 残差连接 [cite: 53]
        x_residual = x + self.dropout(sublayer_output)
        
        # 3. Norm: Layer Normalization [cite: 53]
        return self.norm(x_residual)

# ---------------------------------------------------------------
# (选做) 组合：构建一个完整的 Encoder Layer
# ---------------------------------------------------------------
# 虽然 60-70 分档没有明确要求 ，但 70-80 分档提到了 "encoder block" 。
# 我们可以把上面的组件组合起来，这有助于写报告和后续实验。

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # 1. 多头自注意力 + Add & Norm
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        
        # 2. FFN + Add & Norm
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        
    def forward(self, x, mask=None):
        # MHA 子层
        # 注意：在 self-attention 中, Q, K, V 都是 x
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        
        # FFN 子层
        x = self.sublayer2(x, self.feed_forward)
        
        return x

# ---------------------------------------------------------------
# 测试代码 (确保我们的模块能运行)
# ---------------------------------------------------------------
if __name__ == "__main__":
    # 使用作业中的示例超参数 [cite: 80]
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2 # 作业示例是 2 层 [cite: 80]
    batch_size = 32
    seq_len = 10 # 假设序列长度为 10
    dropout = 0.1
    
    # 0. 准备假数据
    # (batch_size, seq_len, d_model)
    dummy_input = torch.rand(batch_size, seq_len, d_model)
    
    # 1. 测试 PositionalEncoding
    pe = PositionalEncoding(d_model, dropout, max_len=50) # <-- 传入 dropout
    pe_output = pe(dummy_input)
    print(f"PositionalEncoding - Input shape: {dummy_input.shape}")
    print(f"PositionalEncoding - Output shape: {pe_output.shape}\n")
    
    # 2. 测试 MultiHeadAttention (作为自注意力)
    mha = MultiHeadAttention(d_model, num_heads)
    # Q, K, V 都是 pe_output
    mha_output, weights = mha(pe_output, pe_output, pe_output, mask=None)
    print(f"MultiHeadAttention - Output shape: {mha_output.shape}")
    print(f"MultiHeadAttention - Weights shape: {weights.shape}\n")
    
    # 3. 测试 FFN
    ffn = PositionWiseFeedForward(d_model, d_ff)
    ffn_output = ffn(mha_output)
    print(f"PositionWiseFeedForward - Output shape: {ffn_output.shape}\n")
    
    # 4. 测试 EncoderLayer (组合)
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
    encoder_output = encoder_layer(pe_output, mask=None)
    print(f"EncoderLayer - Output shape: {encoder_output.shape}\n")
    
    print("所有模块测试通过！")