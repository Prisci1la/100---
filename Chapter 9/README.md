# Chapter 9

## Files

- `knock74.py` - 正解率的计算
- `knock75.py` - Padding处理
- `knock76.py` - Mini-batch学习
- `knock77.py` - GPU上的学习
- `knock78.py` - 单词嵌入的学习
- `knock79.py` - 架构的修改
- `chapter9_utils.py` - 第9章共用工具

## Example

```bash
python knock74.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000 --epochs 5
python knock75.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000 --batch-size 4
python knock76.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000 --epochs 5 --batch-size 64
python knock77.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000 --epochs 5 --batch-size 256
python knock78.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000 --epochs 5 --batch-size 256
python knock79.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000 --epochs 5 --batch-size 256
```

SST-2数据默认复用 `Chapter 8/data/SST-2`，词向量默认读取 `Chapter 6/GoogleNews-vectors-negative300.bin`。
