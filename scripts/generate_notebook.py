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
    cells = [cell("markdown", """# Qwen3-1.7B → RK3588 Vulkan：官方权重量化构建与验证

本 Notebook 基于固定版本的官方 BF16 权重构建 `Q4_K_M` GGUF 模型，通过双精度配对指令测试与小语料困惑度（PPL）评估核验量化质量，并将校验通过的产物归档至 Google Drive。

- **前置条件**：运行前须在 Colab 界面手动挂载 Google Drive（流水线禁用自动弹窗挂载）。实例需满足至少 **15 GiB 本地可用磁盘** 与 **6 GiB 可用内存**（`MemAvailable`）。建议使用 CPU 运行时，量化与验证流程不依赖 GPU。
- **执行机制**：按序执行全部单元格。支持断点重跑，已完成且通过 SHA256 核验的本地检查点与 Drive 缓存将自动复用。流水线采用失败即停策略：若前置阶段异常，后续单元格将拒绝执行；排查修复后需从失败阶段重新运行。最终成功状态以全部 8 个阶段的完成标记及 `model_preparation_complete` 状态为准，不应仅凭进程退出码判断。
- **工程边界**：本 Notebook 为自包含实现（与 `scripts/colab_prepare.py` 同源），不依赖外部本地文件，亦不发起任何面向编译服务器或 RK3588 开发板的网络连接。Colab CPU 验证不替代板端 Vulkan 算子验证，小样本诊断亦不代表无损量化。
""", 0), cell("code", "# @title 加载完整准备代码（不运行任务）\n" + embedded, 1)]
    sections = [
        ("1. 锁定版本与超参数", "初始化工作目录，锁定模型 Revision、工具链提交哈希、随机种子与评估语料指纹。", "stage1_config()"),
        ("2. 校验运行环境", "核验 Google Drive 挂载状态、本地可用磁盘容量与可用物理内存阈值，不满足则立即终止。", "stage2_environment()"),
        ("3. 核验并恢复缓存", "仅当配置指纹、完成标记、评估报告绑定关系及全部文件 SHA256 均匹配时复用缓存。", "stage3_cache()"),
        ("4. 编译原生 CPU 工具链", "检出固定提交的 `llama.cpp` 源码并编译原生 CPU 工具；使用 `llama-completion` 验证官方聊天模板与多轮上下文。", "stage4_tools()"),
        ("5. 获取官方 BF16 权重", "拉取指定 Revision 的官方 `safetensors` 权重与配置并逐文件校验哈希；有效缓存命中时跳过。", "stage5_download()"),
        ("6. 转换 BF16 GGUF 与 Q4_K_M 量化", "中间态 BF16 文件仅留存于本地运行时，完整输出经元数据与哈希核验后写入本地检查点。", "stage6_convert_quantize()"),
        ("7. 双精度配对评估", "在 BF16 与 Q4_K_M 下依次执行 10 项中文指令样例与 4×`512`-token 窗口 PPL 评估（实际计分 **1020 个 next-token**）。", "stage7_validate()"),
        ("8. 归档至 Drive 并回读校验", "将 `Q4_K_M` 产物、元数据清单、评估报告与日志写入 Drive，完成回读哈希核验后生成 `COMPLETE.json`。", "stage8_publish()"),
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
