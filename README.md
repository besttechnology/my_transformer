# 创 建conda环 境
 $ conda create -n transformer_hw python=3.10
 $ conda activate transformer_hw

# 安 装 依 赖
 $ pip install torch torchtext spacy matplotlib numpy
 $ python -m spacy download en_core_web_sm
 $ python -m spacy download de_core_news_sm

 # 设 置 随 机 种 子 保 证 可 复 现 性
 $ export PYTHONHASHSEED=42
 $ export CUBLAS_WORKAROUND_ROUNDING_MODE=1

 # 运 行 训 练 （包 含 完 整 参 数 设 置）
 $ python train.py \

 --batch_size 32 \
 --epochs 20 \
 --learning_rate 3e-4 \
 --d_model 128 \
 --num_heads 4 \
 --num_layers 2 \
 --d_ff 512 \
 --dropout 0.1 \
 --seed 42
