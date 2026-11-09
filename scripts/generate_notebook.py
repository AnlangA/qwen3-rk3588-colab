#!/usr/bin/env python3
"""Generate a fully self-contained Notebook from colab_prepare.py."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/colab_prepare.py"
DESTINATION = ROOT / "notebooks/Qwen3_RK3588_GPU_Prepare.ipynb"


def cell(kind, text, index):
    result = {"cell_type": kind, "id": f"qwen3-{index:02d}", "metadata": {},
              "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    script = SOURCE.read_text(encoding="utf-8")
    embedded = script.split('\nif __name__ == "__main__":', 1)[0]
    cells = [cell("markdown", """# Qwen3-1.7B → RK3588 Vulkan：官方权重量化准备

此 Notebook 从固定版本的官方 BF16 权重转换并量化 `Q4_K_M`，随后比较基本中文输出和小语料 PPL，并将已验证产物缓存到 Google Drive。

- **运行前**：在 Colab 界面手动挂载 Google Drive。自动执行不会弹出挂载交互。需要至少 **15 GiB 本地磁盘** 和 **6 GiB 可用内存**。优先使用 CPU；不需要为了量化申请 GPU。
- **执行方式**：依次运行全部单元格；中断后可重新运行，完整缓存和完整转换检查点会经 SHA256 核验后复用。前一步失败后，后续单元格会拒绝执行；修复后先重跑失败步骤。不要仅根据 Colab CLI 的退出码判断成功，最后必须看到八个成功标记及 `model_preparation_complete`。代码完整嵌入，无需本地 Mac 文件。Python 脚本与本 Notebook 使用同一份实现。
- **范围**：本 Notebook 不连接 Ubuntu 服务器或开发板，不部署或修改显示服务。Colab CPU 验证不代表板端 Vulkan 验证；小样本验证不代表量化无损。
""", 0), cell("code", "# @title 加载完整准备代码（不运行任务）\n" + embedded, 1)]
    sections = [
        ("1. 固定版本与参数", "模型来源、工具版本和验证样例固定，不自动跟随 `main`。", "stage1_config()"),
        ("2. 检查运行环境", "Drive 未挂载或资源不足将立即停止。不会自动创建运行时或交互挂载。", "stage2_environment()"),
        ("3. 检查并恢复缓存", "仅接受版本、参数、完成标记和文件 SHA256 全部匹配的缓存。", "stage3_cache()"),
        ("4. 准备原生 CPU 工具", "使用固定 `llama.cpp` 源码。额外的 `llama-completion` 用于准确验证官方模板和完整多轮历史。", "stage4_tools()"),
        ("5. 下载官方 BF16 权重", "仅下载固定 revision 的官方权重和配置；有效缓存命中时跳过。", "stage5_download()"),
        ("6. 转换 BF16 GGUF 并量化 Q4_K_M", "中间 BF16 文件仅保留在运行时本地，默认不复制到 Drive。输出完成后才提交检查点。", "stage6_convert_quantize()"),
        ("7. 配对验证", "两种精度依次运行十个中文样例和四个 `512`-token PPL 分块。PPL 实际计分为 **1020 个 next-token**；报告区分输入和计分 token。有限 PPL 与正常输出只说明这组诊断通过。", "stage7_validate()"),
        ("8. 写入 Drive 并回读核验", "缓存 `Q4_K_M`、来源清单、质量报告和日志，最后写完成标记。此后仍须完成服务器交叉编译，并在部署 Gate 等待明确确认。", "stage8_publish()"),
    ]
    for number, (title, explanation, call) in enumerate(sections, 1):
        cells.append(cell("markdown", f"## {title}\n\n{explanation}\n", len(cells)))
        cells.append(cell("code", f"# @title {title}\nrun_stage({number})\n", len(cells)))
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"},
                             "colab": {"name": DESTINATION.name, "provenance": []},
                             "qwen_prepare_source_sha256": hashlib.sha256(script.encode()).hexdigest()},
                "cells": cells}
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
