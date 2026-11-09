# Qwen3-1.7B GGUF 量化构建与基线验证（面向 RK3588 Vulkan 部署）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnlangA/qwen3-rk3588-colab/blob/main/notebooks/Qwen3_RK3588_GPU_Prepare.ipynb)

本仓库提供面向 RK3588（Mali-G610 / PanVK）端侧部署的 **Qwen3-1.7B** 模型量化流水线。流水线基于 Google Colab CPU 环境，从上游官方 **BF16** 权重构建 **Q4_K_M GGUF** 产物，并执行双精度（BF16 与 Q4_K_M）配对功能测试、困惑度（PPL）评估、SHA256 完整性校验及 Google Drive 持久化归档。

[`notebooks/Qwen3_RK3588_GPU_Prepare.ipynb`](notebooks/Qwen3_RK3588_GPU_Prepare.ipynb) 内嵌完整代码与评估语料，支持单文件独立运行；[`scripts/colab_prepare.py`](scripts/colab_prepare.py) 为同源命令行入口。参考运行记录与评估报告归档于 [`examples/`](examples/)。

## 版本锁定与构建基准

| 配置项 | 设定值 |
| :--- | :--- |
| 基座模型 | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |
| 模型 Revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `llama.cpp` Commit | [`d6be065572cdb7bf64014c8c5771e210c9570e44`](https://github.com/ggml-org/llama.cpp/tree/d6be065572cdb7bf64014c8c5771e210c9570e44) |
| 量化路径 | 官方 BF16 (`safetensors`) → BF16 GGUF → Q4_K_M GGUF（不引入 importance matrix） |
| 验证配置 | 原生 CPU 推理；官方 Tokenizer 聊天模板（`enable_thinking=False`） |

**可复现性说明**：上游模型权重与 `llama.cpp` 源码提交均已严格锁定；Python 转换依赖依据该提交的 `requirements` 安装，并在构建时导出 `pip freeze` 快照。由于底层宿主机的传递依赖未做全量锁死，跨环境重复构建不保证产出逐字节一致（bit-for-bit identical）的 GGUF 文件；本文档记录的 SHA256 摘要作为参考构建产物的唯一标识。

## 运行环境与执行流程

### 环境要求

- **运行时**：Google Colab CPU 实例（无需分配 GPU）
- **系统资源**：Python ≥ 3.10，本地可用磁盘空间 ≥ **15 GiB**，可用物理内存（`MemAvailable`）≥ **6 GiB**
- **外部依赖**：可访问 GitHub 与 Hugging Face，具备 `git`、`cmake` 及 C++ 编译工具链（Colab 默认预装）

可在单元格中执行以下命令核验编译工具链：

```bash
!git --version
!cmake --version
!g++ --version
```

### 执行步骤

1. 点击页首 **Open In Colab** 打开 Notebook，切换至 CPU 运行时。
2. 预先挂载 Google Drive（通过 Colab 左侧文件面板挂载，或在独立单元格中执行交互式授权）：

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. 顺序执行全部单元格。首次完整运行涵盖源码编译、权重拉取、格式转换、双精度推理与 PPL 评估，总耗时取决于实例 CPU 性能与网络带宽。
4. 终态核验：确认输出包含全部 8 个阶段的 `Stage N/8 passed` 标记，且最终状态为 `model_preparation_complete`。不应仅凭执行器进程退出码（exit code 0）判定构建成功。

> [!IMPORTANT]
> 流水线采用非交互式失败即停（fail-fast）策略。若 Google Drive 未挂载、硬件资源低于阈值或任一阶段异常，流程将立即终止；脚本不包含交互式挂载或远端设备（服务器/开发板）网络操作。

## 流水线阶段定义

| 阶段 | 入口函数 | 核心操作与校验逻辑 |
| :---: | :--- | :--- |
| 1 | `stage1_config` | 初始化工作目录，锁定模型版本、工具链提交哈希、随机种子与评估超参数 |
| 2 | `stage2_environment` | 校验 Google Drive 挂载状态、本地可用磁盘容量与 `MemAvailable` 内存阈值 |
| 3 | `stage3_cache` | 校验远端缓存的配置指纹、文件体积、SHA256 清单、评估报告绑定关系及完成标记 |
| 4 | `stage4_tools` | 检出固定提交的 `llama.cpp`，编译原生 CPU 工具（`llama-quantize`、`llama-cli`、`llama-completion`、`llama-perplexity`）并配置转换依赖 |
| 5 | `stage5_download` | 下载指定 Revision 的官方 `safetensors` 权重与配置文件，逐文件核验 SHA256 与模型架构参数 |
| 6 | `stage6_convert_quantize` | 执行 BF16 GGUF 转换与 Q4_K_M 量化，核验元数据（架构、28 层 block、词表规模及张量类型分布） |
| 7 | `stage7_validate` | 在 BF16 与 Q4_K_M 下配对执行 10 项中文指令样例与 4×512-token 语料 PPL 评估 |
| 8 | `stage8_publish` | 将 Q4_K_M 产物、清单、评估报告与日志写入 Drive，完成回读哈希核验后写入 `COMPLETE.json` |

为控制存储开销，原始 BF16 权重与中间态 BF16 GGUF 仅在本地运行时暂存；Google Drive 仅持久化经校验的 Q4_K_M 模型、元数据清单（`manifest.json`）、质量报告、`SHA256SUMS`、冒烟测试输入及运行日志。

## 目录结构与命令行接口

默认路径配置如下：

```text
/content/qwen3-rk3588-work/                  # 本地临时工作目录
/content/drive/MyDrive/qwen3-rk3588-cache/
  └── <settings-fingerprint-24chars>/        # 基于配置与语料哈希派生的隔离缓存目录
```

在已挂载 Drive 的环境中，可通过命令行直接调用构建脚本：

```bash
!git clone https://github.com/AnlangA/qwen3-rk3588-colab.git /content/qwen3-rk3588-colab
!python /content/qwen3-rk3588-colab/scripts/colab_prepare.py
```

如需自定义工作或缓存路径，须在启动脚本或执行首个单元格前注入环境变量：

```python
import os
os.environ['QWEN_WORK_DIR'] = '/content/my-qwen3-work'
os.environ['QWEN_DRIVE_ROOT'] = '/content/drive/MyDrive'
```

支持通过 `--start-stage N --end-stage M` 指定执行区间（需保留原工作目录中的 `run_state.json` 及全部前置阶段完成记录）：

```bash
python scripts/colab_prepare.py --start-stage 6 --end-stage 8
```

**断点恢复与缓存策略**：

- **本地检查点**：重跑时自动校验本地中间产物的字节大小与 SHA256，校验通过则直接复用，未完成的分片（`.partial`）将被清理重建。
- **远端缓存命中**：当 Drive 缓存通过完整性校验时，流水线跳过官方权重下载与 BF16 配对评估，仅编译本地工具、核验 GGUF 结构并执行单次 Q4_K_M 冒烟推理。
- **容错与隔离**：损坏或配置不匹配的缓存将被拒绝并触发全量重建，不回退至第三方预量化权重。任一阶段异常均会阻断后续阶段，修复后须从失败阶段重新执行。

## 基线评估结果与方法学边界

| 指标 / 属性 | 记录值 |
| :--- | :--- |
| 产物文件名 | `Qwen3-1.7B-Q4_K_M.gguf` |
| 文件大小 | **1,282,439,520 bytes**（1.28 GB / 1.19 GiB） |
| SHA256 摘要 | `ee582cfa297a5a385412582d25d8fb2f8e565125ab6ea6cddf9e3d121afc261d` |
| BF16 困惑度（PPL） | `24.9287` |
| Q4_K_M 困惑度（PPL） | `26.7812` |
| PPL 相对增量 | `+7.43%` |

**评估协议**：

- **功能样例**：涵盖摘要、信息提取、英中翻译、格式约束、算术计算、数值排序、语体改写、对比分析、多轮状态追踪与概念解释共 10 类任务（解码参数：`seed=42`、`temperature=0`、`max_new_tokens=128`）。
- **困惑度计算**：截取自建诊断语料的前 4 个完整上下文窗口（`n_ctx=512`，共 `2048` 输入 token）；受 `llama-perplexity` 窗口前半段掩码机制决定，实际参与计分的预测位置为 **1020 个 next-token**。语料哈希、token 统计与双精度逐题输出详见 [`examples/quality_report.json`](examples/quality_report.json)。

> [!NOTE]
> **评估局限性**：上述小样本测试与诊断语料仅用于排查转换与量化过程中的结构性退化，**不构成通用能力基准或无损量化证明**。实验表明，上游原始 BF16 模型在算术与排序样例中已存在固有推理偏差。

针对上游 BF16 模型的算术与排序偏差，仓库提供独立对照脚本 [`scripts/check_hf_reference.py`](scripts/check_hf_reference.py)，用于在保留本地 `safetensors` 与中间文件的条件下，直接比对 Hugging Face `transformers`（BF16）与 `llama.cpp` 的词元化序列及贪婪解码输出：

```bash
/content/qwen3-rk3588-work/.venv/bin/python /content/qwen3-rk3588-colab/scripts/check_hf_reference.py
```

**工程边界**：Colab CPU 上的数值验证不能替代目标硬件（RK3588 Mali-G610 / PanVK）上的算子兼容性测试。本仓库职责严格限定于模型准备阶段；交叉编译、Slint 图形界面集成、系统服务封装及板端部署均在下游独立工程中完成。

## 开发验证

```bash
# 由 scripts/colab_prepare.py 单一源码同步生成自包含 Notebook
python3 scripts/generate_notebook.py

# 运行离线单元测试（验证缓存校验、状态机阻断、断点恢复及代码内嵌一致性）
python3 -m unittest discover -s tests -v
```

本次参考构建的环境参数与校验基准记录于 [`examples/preparation-reference.json`](examples/preparation-reference.json)。本仓库遵循最小化追踪原则，不纳入模型权重、云端私有路径、凭据或板端连接配置；所有构建产物（`.gguf`、`.safetensors`）、虚拟环境与运行日志均已通过 [`.gitignore`](.gitignore) 排除。

## 许可协议

本仓库原创代码、Notebook 与诊断语料基于 [MIT License](LICENSE) 开源。流程所引用的预训练模型权重、Tokenizer、`llama.cpp` 及相关第三方依赖均遵循各自上游许可协议。
