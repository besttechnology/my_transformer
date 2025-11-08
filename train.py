import sys
import os
# 将项目根目录 (train.py 所在的目录) 添加到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import time

from src.model import Transformer
from data_setup import get_dataloaders, PAD_IDX

# --- 1. 设置超参数 (来自作业 PDF) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 模型参数

D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
NUM_LAYERS = 2 # 作业建议 2 层
DROPOUT = 0.1

# 训练参数
BATCH_SIZE = 32 # 作业建议 32
LEARNING_RATE = 3e-4 # 作业建议 3e-4
EPOCHS = 5 # 先跑5个epoch看看效果

# --- 2. 准备数据和模型 ---
train_loader, valid_loader, vocab_en, vocab_de = get_dataloaders(BATCH_SIZE)
SRC_VOCAB_SIZE = len(vocab_en)
TGT_VOCAB_SIZE = len(vocab_de)

model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    dropout=DROPOUT
).to(DEVICE)

# 统计参数 (进阶要求)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型总参数量: {count_parameters(model):,}")

# --- 3. 定义优化器和损失函数 ---
# 进阶要求：AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

# 进阶要求：学习率调度 (使用 "Attention is All You Need" 中的调度器)
def lr_lambda(step):
    d_model_tensor = torch.tensor(D_MODEL, dtype=torch.float32)
    step_num = torch.tensor(step, dtype=torch.float32)
    warmup_steps = torch.tensor(4000, dtype=torch.float32)
    
    arg1 = step_num.pow(-0.5)
    arg2 = step_num * (warmup_steps.pow(-1.5))
    
    return d_model_tensor.pow(-0.5) * torch.min(arg1, arg2)

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# 损失函数 (忽略 padding)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# --- 4. 训练和验证循环 ---

def train_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train() # 设置为训练模式
    total_loss = 0
    start_time = time.time()
    
    for i, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        # --- 关键的 Seq2Seq 移位 ---
        # Decoder 的输入是 <bos>...<eos> (去掉末尾)
        tgt_input = tgt[:, :-1]
        # 预测的目标是 ...<eos> (去掉开头 <bos>)
        tgt_output = tgt[:, 1:]
        
        # 创建掩码
        src_mask, tgt_mask = model.create_masks(src, tgt_input, PAD_IDX, DEVICE)
        
        # 前向传播
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # 计算 Loss (基础要求：loss下降)
        # (N, T, C) -> (N*T, C)
        # (N, T) -> (N*T)
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]), 
            tgt_output.reshape(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 进阶要求：梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step() # 更新学习率
        
        total_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
    end_time = time.time()
    epoch_loss = total_loss / len(dataloader)
    print(f"--- 训练 Epoch 耗时: {end_time - start_time:.2f}s ---")
    return epoch_loss

def evaluate(model, dataloader, criterion):
    model.eval() # 设置为评估模式
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = model.create_masks(src, tgt_input, PAD_IDX, DEVICE)
            
            logits = model(src, tgt_input, src_mask, tgt_mask)
            
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]), 
                tgt_output.reshape(-1)
            )
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

# --- 5. 启动训练 ---
print("开始训练...")
train_losses = []
valid_losses = []

for epoch in range(1, EPOCHS + 1):
    print(f"\n======= Epoch {epoch}/{EPOCHS} =======")
    
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    valid_loss = evaluate(model, valid_loader, criterion)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f"\nEpoch {epoch} 结果:")
    print(f"\t训练 Loss: {train_loss:.4f}")
    print(f"\t验证 Loss: {valid_loss:.4f} (我们期望这个值在下降)")
    
    # 进阶要求：模型保存
    torch.save(model.state_dict(), f"transformer_epoch_{epoch}.pt")
    print(f"模型已保存为 transformer_epoch_{epoch}.pt")

print("\n训练完成。")

# 进阶要求：训练曲线可视化 (简单打印)
print("\n训练曲线 (Loss):")
print("Epoch | Train Loss | Valid Loss")
for i in range(EPOCHS):
    print(f"{i+1:5d} | {train_losses[i]:10.4f} | {valid_losses[i]:10.4f}")