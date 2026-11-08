# Qwen3-1.7B Colab 量化准备（RK3588 Vulkan 项目）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnlangA/qwen3-rk3588-colab/blob/main/notebooks/Qwen3_RK3588_GPU_Prepare.ipynb)

从 **Qwen 官方 BF16 权重**自行生成 **Q4_K_M GGUF**，在 Colab CPU 上完成转换、配对诊断、SHA256 校验和 Google Drive 缓存。本仓库保存 RK3588 Mali-G610 / PanVK 聊天项目本次实际使用的模型准备脚本。

Notebook 完整内嵌 Python 实现、中文样例和语料，可独立运行；`scripts/colab_prepare.py` 是同一份实现的命令行入口。仓库中的 Notebook 清除了执行输出，本次结果见 [`examples/`](examples/)。

## 固定版本

| 项目 | 版本 |
|---|---|
| 官方模型 | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |
| 模型 revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| llama.cpp commit | [`d6be065572cdb7bf64014c8c5771e210c9570e44`](https://github.com/ggml-org/llama.cpp/tree/d6be065572cdb7bf64014c8c5771e210c9570e44) |
| 转换路线 | 官方 BF16 → BF16 GGUF → Q4_K_M，无额外 importance matrix |
| 推理验证 | CPU；官方 tokenizer 模板，`enable_thinking=False` |

模型和 llama.cpp 已固定；Python 转换依赖从该 llama.cpp 版本的 requirements 安装，并记录 `pip freeze`。**整个 Python 环境没有完整锁定**，重跑不保证得到相同字节的 GGUF；本次文件哈希用作本次产物的身份记录。

## 在 Colab 运行

1. 点击上方 **Open In Colab**，按需选择“在云端硬盘中保存副本”。选择 CPU 运行时即可，脚本不使用 GPU。
2. 先通过 Colab 文件面板手动挂载 Google Drive。也可在一个单独单元格执行下面的代码，按提示授权：

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. 确认 Python ≥3.10、至少 **15 GiB 本地可用磁盘**和 **6 GiB `MemAvailable`**；需要能访问 GitHub/Hugging Face，以及 Git、CMake、C++ 编译器。Colab 通常已提供这些构建工具，可先检查：

   ```python
   !git --version
   !cmake --version
   !g++ --version
   ```

4. 依次运行 Notebook 全部单元格。首次运行包含源码构建、权重下载、BF16 推理和 PPL，耗时随 Colab CPU、网络变化，不承诺固定完成时间。
5. 最后确认八个 `Stage N/8 passed` 标记齐全，且状态为 `model_preparation_complete`。仅看到 Notebook 执行器退出成功，不能证明准备成功。

**Drive 未挂载、资源不足或任一步出错时会停止。**自动流程不会触发交互式挂载，也不会连接服务器或开发板。

## 八个阶段

| 阶段 | 行为 |
|---|---|
| 1 | 固定模型、工具、seed 和测试参数 |
| 2 | 检查 Drive、内存和磁盘 |
| 3 | 校验缓存版本、SHA256、质量报告和完成标记 |
| 4 | 构建 CPU 原生 `llama-quantize`、`llama-cli`、`llama-completion`、`llama-perplexity`，安装转换依赖 |
| 5 | 下载固定 revision 官方权重、tokenizer 和配置，记录文件校验 |
| 6 | 先转换 BF16 GGUF，再量化 Q4_K_M，检查架构、28 层、词表和张量类型 |
| 7 | BF16/Q4 依次运行同一组中文功能样例和小语料 PPL |
| 8 | 写入 Drive、回读校验，最后写完成标记 |

BF16 权重、中间 BF16 GGUF 和模型加载按顺序处理，默认只在运行时本地保留。长期缓存保存 Q4_K_M、manifest、质量报告、SHA256 清单、冒烟提示和日志。

## 目录与命令行用法

默认目录：

```text
/content/qwen3-rk3588-work/                 # 本地工作目录
/content/drive/MyDrive/qwen3-rk3588-cache/
  <settings-fingerprint前24位>/            # 按版本、参数及语料区分
```

也可在已挂载 Drive 的 Colab 单元格中使用脚本：

```python
!git clone https://github.com/AnlangA/qwen3-rk3588-colab.git /content/qwen3-rk3588-colab
!python /content/qwen3-rk3588-colab/scripts/colab_prepare.py
```

如需自定义目录，必须在执行脚本或 Notebook 第一个代码单元格之前设置：

```python
import os
os.environ['QWEN_WORK_DIR'] = '/content/my-qwen3-work'
os.environ['QWEN_DRIVE_ROOT'] = '/content/drive/MyDrive'
```

脚本支持 `--start-stage N --end-stage M`。从中间阶段恢复需要原工作目录的 `run_state.json` 及全部前置阶段成功标记，例如：

```bash
python scripts/colab_prepare.py --start-stage 6 --end-stage 8
```

**重跑与缓存：**可从第 1 阶段重新运行；完整转换检查点经过大小和 SHA256 校验后复用，未完成临时文件重新生成。完整 Drive 缓存命中后仍构建工具、检查 GGUF 并执行一次 Q4 冒烟，跳过官方权重下载和重新 BF16 配对诊断。损坏缓存不复用，重新从官方权重生成，不改用第三方量化文件。失败阶段会阻止后续阶段执行，修复后先重跑失败阶段。

## 本次量化结果与限制

| 项目 | 记录 |
|---|---|
| 输出 | `Qwen3-1.7B-Q4_K_M.gguf` |
| 精确体积 | **1,282,439,520 bytes**（约 1.28 GB / 1.19 GiB） |
| SHA256 | `ee582cfa297a5a385412582d25d8fb2f8e565125ab6ea6cddf9e3d121afc261d` |
| BF16 PPL | 24.9287 |
| Q4_K_M PPL | 26.7812 |
| PPL 相对变化 | 约 +7.43% |

中文样例涵盖摘要、提取、翻译、格式、计算、排序、改写、比较、多轮和简短解释。固定 seed42、temperature0、最多128个新 token。原创小语料使用前四个完整512-token块，共2048个输入 token；该 PPL 实现实际计分 **1020 个 next-token**。语料哈希、实际计数和两种精度的原始输出见 [`examples/quality_report.json`](examples/quality_report.json)。

这些是诊断样例，**不代表量化无损或通用准确率**。原始 BF16 同样出现过计算/排序错误。可选的 [`scripts/check_hf_reference.py`](scripts/check_hf_reference.py) 用本地官方 safetensors 对照这两个问题，并比较 Hugging Face/llama.cpp 的输入 token。它是额外诊断，需先完成主流程、保留 BF16 权重/中间文件，并使用默认 `/content/qwen3-rk3588-work` 目录：

```bash
/content/qwen3-rk3588-work/.venv/bin/python /content/qwen3-rk3588-colab/scripts/check_hf_reference.py
```

Colab 的 CPU 验证不替代板端 Vulkan 测试。本仓库只包含模型准备；RK3588 上的交叉编译、Slint 界面、系统服务和部署属于后续独立流程。

## 修改与检查

```bash
# 从同一份 Python 源码生成干净 Notebook
python3 scripts/generate_notebook.py

# 无需下载模型，检查缓存、失败保护、恢复和 Notebook 嵌入等逻辑
python3 -m unittest discover -s tests -v
```

准备源码与本次记录的 SHA256 见 [`examples/preparation-reference.json`](examples/preparation-reference.json)。仓库不上传模型权重、Drive 私有链接、SSH 凭据或板端连接配置。生成的 `.gguf`、`.safetensors`、环境及日志已加入 `.gitignore`。

## 许可

本仓库原创脚本、Notebook 和诊断语料采用 [MIT License](LICENSE)。下载的模型、tokenizer、llama.cpp 和其他依赖仍遵循各自上游许可。
