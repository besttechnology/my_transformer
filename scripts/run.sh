#!/bin/bash

# set -e: 确保脚本在任何命令执行失败时立即退出
set -e

echo "开始训练..."

# -u: 强制 Python 使用非缓冲的 stdout/stderr，这样日志可以实时写入
# 2>&1: 将标准错误 (stderr) 重定向到标准输出 (stdout)
# | tee train.log: 
#    使用管道符(|)将 stdout 传递给 tee 命令
#    tee 会做两件事：
#    1. 将内容打印到控制台
#    2. 将内容写入 train.log 文件
python -u train.py 2>&1 | tee train.log

echo "训练完成。"