# VSCode Codex AIX 远程开发说明

本文档用于让 VSCode 中的 Codex 插件了解当前 AIX GPU cluster 远程开发环境。

## 1. 当前远程服务器

已成功配置并测试 SSH 连接：

```bash
ssh aix-g20
```

该连接会登录到：

```text
HostName: g20.aix.uec.ac.jp
User: u2630018
OS: Ubuntu 20.04.4 LTS
```

本地 SSH 配置文件位置：

```text
/Users/priscilla/.ssh/config
```

其中应包含：

```sshconfig
Host aix-g20
  HostName g20.aix.uec.ac.jp
  User u2630018
  IdentityFile ~/.ssh/aix_gpu_key
  IdentitiesOnly yes
```

本地私钥位置：

```text
/Users/priscilla/.ssh/aix_gpu_key
```

远程公钥已添加到 AIX 服务器的：

```text
~/.ssh/authorized_keys
```

## 2. VSCode Remote-SSH 使用方式

在 VSCode 中：

1. 打开 Command Palette：`Cmd + Shift + P`
2. 选择 `Remote-SSH: Connect to Host...`
3. 选择 `aix-g20`
4. 连接成功后打开远程目录，例如：

```text
/home/u2630018
```

或项目目录，例如：

```text
/home/u2630018/project
```

## 3. Codex 插件可执行的常用操作

Codex 可以通过本地 SSH 配置访问远程服务器，例如：

```bash
ssh aix-g20 "hostname"
ssh aix-g20 "pwd"
ssh aix-g20 "nvidia-smi"
```

如果需要在远程项目中运行命令：

```bash
ssh aix-g20 "cd ~/project && python train.py"
```

如果需要打开远程交互 shell：

```bash
ssh aix-g20
```

## 4. 本地代码同步到 AIX

推荐使用 `rsync` 从本地同步代码到远程。

在本地项目目录执行：

```bash
rsync -av ./ aix-g20:~/project/
```

常见排除项：

```bash
rsync -av \
  --exclude .git \
  --exclude __pycache__ \
  --exclude .venv \
  --exclude node_modules \
  ./ aix-g20:~/project/
```

也可以使用 `scp`：

```bash
scp -r /path/to/local/project aix-g20:~/project
```

如果代码托管在 GitHub，也可以在远程服务器上直接：

```bash
git clone <repository-url>
```

## 5. GPU 使用注意事项

使用前请检查 GPU 状态：

```bash
nvidia-smi
```

原则上使用没有计算进程、显存占用较低的 GPU。

如果只想指定某一张 GPU，例如 GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

如果使用多张 GPU，例如 GPU 0 和 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

## 6. Python 环境

AIX 手册建议优先使用系统提供的 Anaconda：

```bash
export PATH=/usr/local/anaconda3/bin:${PATH}
```

如需安装 Python 包，校内网络可能需要 proxy：

```bash
pip install <package-name> --user --proxy=http://proxy.uec.ac.jp:8080/
```

如果使用 conda，可以创建独立环境：

```bash
conda create -n myenv python=3.10
conda activate myenv
```

如果网络访问失败，先设置 proxy：

```bash
export http_proxy=http://proxy.uec.ac.jp:8080/
export https_proxy=http://proxy.uec.ac.jp:8080/
```

## 7. 重要限制

- 不要长时间占用 GPU 显存但不运行计算。
- 使用 JupyterHub 后，如果不再计算，要关闭 kernel。
- 学生文件总量上限约为 500 GB。
- 不要在多人共用 GPU 上抢占他人正在使用的显存。
- 账号 `u2630018` 在 `g20` 上可用，但 `g24` 曾出现本地用户未同步错误：

```text
500 : Internal Server Error
Error in Authenticator.pre_spawn_start: KeyError "getpwnam(): name not found: 'u2630018'"
```

因此当前优先使用：

```text
aix-g20
```

## 8. 管理员联系方式

如果需要使用其他服务器，例如 `g24`、`g00`，但出现用户不存在或无法登录的问题，请联系：

```text
gpu-admin@aix.uec.ac.jp
```

可说明：

```text
UEC-ID: u2630018
g20 には SSH/JupyterHub でログインできますが、
g24 では getpwnam(): name not found が表示されます。
ユーザ作成・同期状況をご確認いただけますでしょうか。
```
