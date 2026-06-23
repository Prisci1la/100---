# Chapter 8

## Files

- `tutorial01_tensors.py` - PyTorch入门[1]张量
- `tutorial02_datasets_dataloaders.py` - PyTorch入门[2]Dataset与DataLoader
- `tutorial03_transforms.py` - PyTorch入门[3]数据变换
- `tutorial04_build_model.py` - PyTorch入门[4]模型构建
- `tutorial05_autograd.py` - PyTorch入门[5]自动微分
- `tutorial06_optimization_loop.py` - PyTorch入门[6]优化循环
- `tutorial07_save_load_model.py` - PyTorch入门[7]模型保存与读取
- `tutorial08_quickstart.py` - PyTorch入门[8]快速开始
- `knock70.py` - 单词嵌入的读取
- `knock71.py` - 数据集的读取
- `knock72.py` - Bag of words模型的构建
- `knock73.py` - 模型的学习
- `chapter8_utils.py` - 第8章共用工具

## Example

```bash
python knock70.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000
python knock71.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000
python knock72.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000
python knock73.py --vector-path ../Chapter\ 6/GoogleNews-vectors-negative300.bin --max-vocab 50000 --epochs 5 --batch-size 32
```

`knock73.py` prints `Using device: cuda` when CUDA is available.

Use `--max-vocab 0` to load the full GoogleNews vector file.
