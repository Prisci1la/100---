#!/bin/bash

# 一键上传到 Git 仓库

cd "$(dirname "$0")" || exit

echo "📝 正在提交更改..."
git add -A

# 检查是否有更改
if git diff-index --quiet HEAD --; then
    echo "✅ 没有新改动，仓库已是最新"
else
    # 生成提交信息（包含时间戳）
    COMMIT_MSG="更新代码 $(date '+%Y-%m-%d %H:%M:%S')"
    git commit -m "$COMMIT_MSG"
fi

echo "📤 正在推送到远程..."
git push origin main

echo "✅ 上传完成！"
