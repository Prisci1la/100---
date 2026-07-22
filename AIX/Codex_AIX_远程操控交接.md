# Codex AIX 远程操控交接

更新时间：2026-07-18 23:32 JST

这份文档用于切换到 Codex 桌面端后继续操作 AIX GPU 服务器。

## Codex 桌面端接收检查

检查时间：2026-07-18 23:40 JST 左右

已确认本地可以通过 SSH 接上远程：

```bash
ssh aix-g20 "hostname && pwd"
```

返回主机为：

```text
g20.aix.uec.ac.jp
```

远程当前没有正在运行的第 12 章训练进程。`tmux list-sessions` 仍只有：

```text
knock
monitor
```

GPU 当前空闲，8 张 TITAN RTX 都约为 `1 MiB / 24576 MiB`，利用率 `0%`。

注意：远程仍存在旧锁目录：

```bash
/home0/y2026/u2630018/100knock/logs/ch12_run.lock
```

但没有对应的第 12 章进程，因此这是 stale lock。重新启动第 12 章前应先删除：

```bash
ssh aix-g20 "bash -lc 'rm -rf ~/100knock/logs/ch12_run.lock'"
```

接收时还看到 `knock` tmux pane 里曾经手动执行过：

```bash
bash remote_watch_after_ch13_run_ch12.sh
```

这次尝试在准备 SST-2 数据阶段曾出现 Hugging Face `datasets` 的 schema/cast 错误，表现为读取 `glue/sst2` 时混到 `ax` schema：

```text
ValueError: Couldn't cast ... because column names don't match
datasets.builder.DatasetGenerationError: An error occurred while generating the dataset
```

不过当前 `~/100knock/logs/ch12_prepare_sst2.log` 显示：

```text
SST-2 ready: /home0/y2026/u2630018/100knock/Chapter 8/data/SST-2
train bytes: 3806081
dev bytes: 94931
```

所以下一次启动可以先删旧锁并直接重跑。如果再次遇到 `datasets` 读取 GLUE 的 schema 错误，优先处理 Hugging Face 缓存，或修改准备 SST-2 的逻辑，让它在 `train.tsv` 和 `dev.tsv` 已存在时完全跳过 `load_dataset("glue", "sst2")`。

## 第 12 章续跑结果

续跑时间：2026-07-18 23:37-23:57 JST 左右

已清理旧锁并启动：

```bash
tmux new-session -d -s ch12 ~/100knock/remote_run_ch12_now.sh
```

结果：

- `knock90.py` 到 `knock96.py` 均完成。
- `knock91.py` 已越过旧的 `Tensor.tile()` 错误，说明 `chapter12_utils.py` 的兼容补丁有效。
- `knock96.py --max-examples 200` 完成，日志显示 `accuracy: 0.735000`。
- `knock97.py --epochs 3 --batch-size 16` 完成：

```text
epoch 01: train_loss=0.420917, dev_accuracy=0.865826
epoch 02: train_loss=0.398262, dev_accuracy=0.869266
epoch 03: train_loss=0.393835, dev_accuracy=0.880734
```

原脚本中的 8 GPU 分布式 `knock98.py` 失败：

```text
RuntimeError: Unsupported data type for NCCL process group
```

尝试在 `knock98.py` 和 `knock99.py` 的 DDP 包装上加 `broadcast_buffers=False` 后，旧 torch 1.5.1 仍在 DDP 初始化阶段报同样 NCCL 错误。因此改用单进程 legacy fallback 续跑。

新增续跑脚本：

```bash
~/100knock/remote_continue_ch12_98_99.sh
```

本地副本：

```bash
AIX/remote_continue_ch12_98_99.sh
```

实际运行命令：

```bash
python knock98.py --epochs 1 --batch-size 2 --gradient-accumulation-steps 8 --max-train-examples 2000
python knock99.py --epochs 1 --batch-size 1 --gradient-accumulation-steps 8 --max-train-examples 1000
```

结果：

- `knock98.py` 单进程 legacy 完成，日志：`logs/ch12_knock98_legacy_single.log`
- `knock99.py` 单进程 legacy 完成，日志：`logs/ch12_knock99_legacy_single.log`
- 两个日志均确认跑到 `100%`，无新的 `Traceback` / `RuntimeError`。
- 输出模型目录：

```bash
~/100knock/Chapter\ 12/models/sft_sentiment_gpt2
~/100knock/Chapter\ 12/models/dpo_sentiment_gpt2
```

两者各约 `1.4G`。GPU 已释放，`nvidia-smi` 显示 8 张 TITAN RTX 基本空闲。

## 当前结论

远程主机：

```bash
ssh aix-g20
```

远程项目目录：

```bash
/home0/y2026/u2630018/100knock
```

远程数据目录实际落在：

```bash
/data/student/u2630018/100knock
```

现在第 12 章没有在运行。`tmux list-sessions` 只看到 `knock` 和 `monitor`，没有 `ch12`。

第 12 章状态：

- `knock90.py` 已完成，输出了 GPT-2 medium 的 next-token 预测结果。
- `knock91.py` 失败，原因为远端 PyTorch 是 `1.5.1`，没有 `Tensor.tile()`，而当前 transformers 生成代码调用了它。
- 已在本地和远端同步补丁：`Chapter 12/chapter12_utils.py` 里加入了 `Tensor.tile()` 兼容实现。
- 补丁同步完成后，还没有重新启动第 12 章训练。

第 13 章状态：

- 已跑完。
- `knock93` BLEU 日志显示 `BLEU: 2.22`。

第 14 章状态：

- 误进入过 `knock95`，后来手动停止。
- 已完成 SentencePiece、subword 数据、subword 训练 5 epoch。
- beam 搜索跑到 `beam=10` 约 54% 时停止。
- `knock96.py` 和 `knock97.py` 未开始。

## SSH 配置

本地 SSH 配置文件：

```bash
/Users/priscilla/.ssh/config
```

当前关键配置：

```sshconfig
Host aix-g20
  HostName g20.aix.uec.ac.jp
  User u2630018
  IdentityFile ~/.ssh/aix_gpu_key
  IdentitiesOnly yes
  ServerAliveInterval 60
  ServerAliveCountMax 5
  TCPKeepAlive yes
```

检查能不能连：

```bash
ssh aix-g20 "hostname"
```

如果 DNS 或连接失败，通常先检查 UEC VPN 是否还连着。

## Codex 桌面端工作流

在 Codex 桌面端打开本地项目：

```bash
/Users/priscilla/Priscilla/Weekly/100ノック
```

之后可以让 Codex 直接执行本地 SSH 命令，例如：

```bash
ssh aix-g20 "bash -lc 'cd ~/100knock && pwd && nvidia-smi'"
```

常用原则：

- 本地改代码，然后用 `rsync` 或 `scp` 上传到 AIX。
- 训练在远端 `tmux` 里跑，不依赖本地 VS Code 窗口。
- VPN/SSH 断开后，远端 `tmux` 里的训练一般还会继续。
- 不要用 `Ctrl+C`，除非明确要停止训练。

## 重新启动第 12 章

先检查有没有旧进程：

```bash
ssh aix-g20 "bash -lc 'tmux list-sessions 2>/dev/null || true; ps -u u2630018 -f | grep -E \"remote_run_ch12|knock9|distributed.launch|python\" | grep -v grep || true'"
```

启动第 12 章：

```bash
ssh aix-g20 "bash -lc 'tmux kill-session -t ch12 2>/dev/null || true; tmux new-session -d -s ch12 /home0/y2026/u2630018/100knock/remote_run_ch12_now.sh; tmux list-sessions | grep ch12'"
```

进入实时终端：

```bash
ssh aix-g20
tmux attach -t ch12
```

退出实时终端但不停止训练：

```text
Ctrl+b
d
```

## 监控进度

看 tmux 当前画面：

```bash
ssh aix-g20 "bash -lc 'tmux capture-pane -t ch12 -p 2>/dev/null | tail -120 || true'"
```

看第 12 章日志：

```bash
ssh aix-g20 "bash -lc 'ls -lh ~/100knock/logs/ch12_* 2>/dev/null || true; for f in ~/100knock/logs/ch12_*.log; do echo ==== \$f; tail -40 \$f; done'"
```

看 GPU：

```bash
ssh aix-g20 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader'
```

看远端目录大小：

```bash
ssh aix-g20 "bash -lc 'du -sh ~/100knock ~/.cache ~/.local 2>/dev/null'"
```

## 同步代码

因为目录名里有空格，`Chapter 12` 需要转义。

同步单个文件：

```bash
rsync -av "Chapter 12/chapter12_utils.py" "aix-g20:/home0/y2026/u2630018/100knock/Chapter\\ 12/chapter12_utils.py"
```

同步第 12 章整个目录，排除模型和缓存：

```bash
rsync -av \
  --exclude "__pycache__" \
  --exclude "models" \
  --exclude "outputs" \
  --exclude "checkpoints" \
  "Chapter 12/" \
  "aix-g20:/home0/y2026/u2630018/100knock/Chapter\\ 12/"
```

## VPN 处理

当前 macOS VPN 服务名：

```text
UEC
```

命令行检查：

```bash
scutil --nc status UEC
```

之前测试结果：

- `scutil --nc stop UEC` 可以断开。
- `scutil --nc start UEC` 没能稳定重新连接，只显示 `Disconnected`。
- 所以目前更可靠的方法是手动在系统设置里重新打开 VPN。

如果使用 UI 自动点击方案：

- 需要把系统设置的 VPN 页面打开在桌面。
- Codex 可以通过桌面控制工具识别按钮并点击。
- 点击前必须再次确认，因为这会改变本机 VPN 状态，可能导致 SSH 暂时中断。
- UI 自动化比命令行脆弱，窗口位置和 macOS 语言界面变化都可能影响成功率。

## 当前第 12 章失败点

失败日志：

```text
AttributeError: 'Tensor' object has no attribute 'tile'
```

原因：

```text
远端 torch == 1.5.1
transformers 生成代码需要 Tensor.tile()
```

已修复：

```text
Chapter 12/chapter12_utils.py
```

加入了兼容逻辑：

```python
if not hasattr(torch.Tensor, "tile"):
    def _tensor_tile(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return self.repeat(*dims)

    torch.Tensor.tile = _tensor_tile
    torch.tile = lambda tensor, *dims: tensor.tile(*dims)
```

下一步建议：

1. 重新启动第 12 章。
2. 如果又遇到 torch 1.5.1 和 transformers 的兼容问题，继续在 `chapter12_utils.py` 做最小兼容补丁。
3. `knock98.py` 和 `knock99.py` 已经改成旧 torch fallback，不依赖 peft/trl 的训练主流程。
