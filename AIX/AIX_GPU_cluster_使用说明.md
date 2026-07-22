# AIX GPU Cluster 使用说明

本文档根据 AIX GPU cluster 利用规则整理，用于 UEC-ID: `u2630018` 用户注册前阅读与日后使用参考。

## 1. 用户注册前确认

- 使用者原则上应为 AIX 所属教师，或 AIX 所属教师研究室的学生、研究生、博士后等。
- UEC-SHIP 相关研究室，包括石桥单元、宫胁单元的教师和学生也可以使用。
- 如果不属于 AIX 或 UEC-SHIP 相关研究室，但希望使用 GPU cluster，需要联系管理员：
  - AIX 副中心长 柳井
  - `gpu-admin@aix.uec.ac.jp`
- 使用 AIX GPU cluster 需要电通大全学 ID，也就是 UEC 账号。

## 2. 基本使用方式

AIX GPU cluster 有两种主要使用方式：

1. 通过浏览器使用 JupyterHub。
2. 通过 SSH 登录使用。

SSH 登录原则上只允许无密码的公钥认证。当前可能暂时允许密码认证，但可能在没有提前通知的情况下停止。

GPU 服务器使用电通大私有 IP，基本上只能从校内访问。如果需要从校外访问，请使用基盘中心提供的 VPN 服务。

数据传输请使用基于公钥认证的 `scp`。

## 3. 使用目的

- 使用目的限定为教育和研究。
- 不允许将 GPU cluster 用于个人兴趣性质的外部竞赛。
- 如果是课程的一部分，例如信息学工房等，则可以使用。

## 4. GPU 使用规则

- 原则上应使用当前没有计算进程的 GPU。
- 不要在已有计算进程的 GPU 上追加运行大量占用 GPU 内存的进程，否则可能导致先运行的进程因内存不足异常结束。
- 每人同时使用 GPU 的上限原则上为：
  - 最多 16 GPU
  - 最多 3 台服务器
- 如果因论文截止等确有必要，并且其他 GPU 空闲，则 48 小时以内的计算可以临时使用：
  - 最多 24 GPU
  - 最多 4 台服务器

UEC-SHIP 相关人员如果只使用 UEC-SHIP 专用服务器，则上述 GPU 数量限制不适用。

## 5. GPU 内存使用注意事项

使用 TensorFlow 或以 TensorFlow 为后端的 Keras 时，默认可能会让一个进程占用所有 GPU 的全部显存。为了避免影响他人，必须在程序开头加入以下设置：

```python
import tensorflow as tf
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=config)
```

请只申请计算所需的 GPU 内存，并在计算结束后释放。

如果在 Jupyter 上运行程序，计算结束后还需要关闭 iPython kernel。否则即使计算负载为 0，也可能继续占用 GPU 内存。

在 JupyterLab 中关闭 kernel 的方法：

1. 点击左侧的 `Running` 标签。
2. 在 `Kernel Sessions` 中找到已经运行结束的 TensorFlow 或 Keras 进程。
3. 点击 `SHUTDOWN`。

如果长时间无计算负载却占用大量 GPU 内存，管理员可能会在未通知的情况下删除相关进程。

## 6. 文件容量限制与数据管理

AIX GPU 服务器专用文件服务器容量为 100 TB。

请避免长期保留大量训练参数文件、snapshot 文件和不再需要的中间文件。训练结束后，只保留必要的模型参数，删除不需要的文件。

每人文件总使用量上限：

- 学生：最大 500 GB
- 教师、博士后：最大 1 TB

超过上限后，写入时可能出现类似以下错误：

```text
書き込みエラー: 入力/出力エラーです
```

出现该情况时，请删除不需要的文件。

毕业后 UEC ID 通常会在 30 天后失效。UEC ID 失效后，服务器上的数据会全部删除。毕业前请自行备份需要的数据。

## 7. 成果发表时的致谢

如果研究使用了 AIX GPU 服务器，并基于该使用成果进行外部发表，需要在论文或报告中加入类似以下致谢：

```text
本研究は，電気通信大学人工知能先端研究センター(AIX)の計算機を利用して実施したものです．
```

发表后，请将论文 PDF 发送至：

```text
gpu-admin@aix.uec.ac.jp
```

AIX 可能会将相关成果记载在活动报告书中。

UEC-SHIP 相关人员如果只使用 UEC-SHIP 专用服务器，则上述致谢规则不适用。

## 8. SSH 公钥认证设置

在登录来源机器，例如研究室 PC 上生成 SSH key：

```bash
ssh-keygen -t rsa
```

显示公钥：

```bash
cat ~/.ssh/id_rsa.pub
```

复制显示出来的公钥内容。

在 AIX GPU 机器的 JupyterHub terminal 中执行：

```bash
cat >> ~/.ssh/authorized_keys
```

粘贴公钥后，按 `Ctrl+D` 结束输入。

然后设置权限：

```bash
chmod 600 ~/.ssh/authorized_keys
```

建议先在可密码登录的教育用计算机 `sol.edu.cc.uec.ac.jp` 上确认可以通过公钥认证登录。

Windows 机器通过 PuTTY 登录的设置可参考 IED 支持页面：

https://www.ied.inf.uec.ac.jp/support/projects/ied-support/wiki/%E3%83%AA%E3%83%A2%E3%83%BC%E3%83%88%EF%BC%88%E6%BC%94%E7%BF%92%E5%AE%A4%E5%A4%96%EF%BC%89%E3%81%8B%E3%82%89%E3%81%AE%E3%82%A2%E3%82%AF%E3%82%BB%E3%82%B9

## 9. Python 环境

使用 Python 时，建议使用 `/usr/local/anaconda3` 中的 Anaconda Python 3.7：

```bash
export PATH=/usr/local/anaconda3/bin:${PATH}
```

标准 Anaconda 大约 500 MB，请尽量避免在自己的 home directory 中安装完整 Anaconda。

如果需要自行安装环境，建议使用：

- miniconda
- uv

服务器中已安装：

- TensorFlow
- Keras
- Chainer
- PyTorch

这些框架不会频繁更新。如果需要较新版本，请使用 `conda create` 创建自己的 conda 虚拟环境。

## 10. Python 模块安装

如需追加安装 Python 模块，请使用用户权限安装到自己的 home 目录：

```bash
pip install {模块名} --user --proxy=http://proxy.uec.ac.jp:8080/
```

## 11. Docker 使用

如果 Anaconda 虚拟环境难以满足需求，也可以使用 Docker。

Docker 使用需要管理员设置 `docker` group 属性。如需使用，请联系管理员。

## 12. 系统信息与联系

服务器 Ubuntu 版本：

```text
Ubuntu 20.04 LTS
```

问题、需求或申请事项请联系：

```text
gpu-admin@aix.uec.ac.jp
```

管理员人数有限，回复和处理可能需要一定时间。
