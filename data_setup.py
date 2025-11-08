from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

# --- 1. 定义特殊 Token 和索引 ---
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# --- 2. 加载数据集和分词器 ---
def load_data_and_tokenizers():
    print("加载 IWSLT2017 (en-de) 数据集...")
    # 加载数据集
    # 这是修改后的代码：
    dataset = load_dataset("iwslt2017", "iwslt2017-en-de", keep_in_memory=True)
    
    # 获取分词器
    tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
    tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')
    
    return dataset, tokenizer_en, tokenizer_de

# --- 3. 构建词表 ---
def build_vocab(dataset, tokenizer_en, tokenizer_de):
    print("构建词表...")
    
    def yield_tokens(data_iter, tokenizer, lang):
        for data in data_iter:
            yield tokenizer(data['translation'][lang])

    # 英文词表
    vocab_en = build_vocab_from_iterator(
        yield_tokens(dataset['train'], tokenizer_en, 'en'),
        min_freq=2,
        specials=special_symbols,
        special_first=True
    )
    vocab_en.set_default_index(UNK_IDX)

    # 德文词表
    vocab_de = build_vocab_from_iterator(
        yield_tokens(dataset['train'], tokenizer_de, 'de'),
        min_freq=2,
        specials=special_symbols,
        special_first=True
    )
    vocab_de.set_default_index(UNK_IDX)
    
    return vocab_en, vocab_de

# --- 4. 数据处理和Dataloader ---

# 辅助函数：将文本转换为 tensor
def text_transform(tokenizer, vocab):
    def wrapper(text):
        tokens = tokenizer(text)
        return torch.tensor([BOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX])
    return wrapper

# Dataloader 的 "collate" 函数
def create_collate_fn(tokenizer_en, tokenizer_de, vocab_en, vocab_de):
    
    transform_en = text_transform(tokenizer_en, vocab_en)
    transform_de = text_transform(tokenizer_de, vocab_de)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for item in batch:
            src_batch.append(transform_en(item['translation']['en']))
            tgt_batch.append(transform_de(item['translation']['de']))
        
        # 补齐
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        return src_batch, tgt_batch
    
    return collate_fn

# --- 5. 主函数：获取 Dataloaders ---
def get_dataloaders(batch_size=32):
    dataset, tokenizer_en, tokenizer_de = load_data_and_tokenizers()
    vocab_en, vocab_de = build_vocab(dataset, tokenizer_en, tokenizer_de)
    
    collate_fn = create_collate_fn(tokenizer_en, tokenizer_de, vocab_en, vocab_de)
    
    train_iter = dataset['train']
    valid_iter = dataset['validation']
    
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, valid_dataloader, vocab_en, vocab_de

if __name__ == "__main__":
    # 测试数据加载
    train_loader, _, vocab_en, vocab_de = get_dataloaders(batch_size=8)
    print(f"\n英文词表大小: {len(vocab_en)}")
    print(f"德文词表大小: {len(vocab_de)}")
    
    # 取一个 batch
    src_batch, tgt_batch = next(iter(train_loader))
    print(f"源 batch 形状: {src_batch.shape}")
    print(f"目标 batch 形状: {tgt_batch.shape}")
    print("数据加载测试通过！")