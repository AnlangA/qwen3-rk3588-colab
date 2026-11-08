#!/usr/bin/env python3
"""Official Qwen3 BF16 -> Q4_K_M preparation; no board/network deployment code.

This file is also embedded in the accompanying self-contained Colab notebook.
Run with Python 3.10+ on a Colab runtime with Google Drive already mounted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import venv

MODEL_REPO = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LLAMA_COMMIT = "d6be065572cdb7bf64014c8c5771e210c9570e44"
MODEL_NAME = "Qwen3-1.7B-Q4_K_M.gguf"
BF16_NAME = "Qwen3-1.7B-BF16.gguf"
WORK = Path(os.environ.get("QWEN_WORK_DIR", "/content/qwen3-rk3588-work"))
DRIVE = Path(os.environ.get("QWEN_DRIVE_ROOT", "/content/drive/MyDrive"))
SOURCE = WORK / "llama.cpp"
HF_DIR = WORK / "hf-model"
BUILD = WORK / "build-native"
ARTIFACTS = WORK / "artifacts"
LOGS = WORK / "logs"
PYTHON = WORK / ".venv" / "bin" / "python"
THREADS = max(1, min(4, os.cpu_count() or 1))
STATE: dict = {}

PROMPTS = [
    {"id": "summary", "messages": [{"role": "user", "content": "用一句话概括：社区图书馆本周新增了两百本图书，并将周末开放时间延长到晚上八点，方便居民下班后阅读。"}]},
    {"id": "extraction", "messages": [{"role": "user", "content": "只返回JSON：小林周六下午三点在科技馆参加机器人活动。提取人物、时间、地点三个字段。"}]},
    {"id": "translation", "messages": [{"role": "user", "content": "把这句话翻译成自然的中文，只输出译文：The meeting has been moved to Friday morning."}]},
    {"id": "format", "messages": [{"role": "user", "content": "列出三种常见水果，每行一个名称，不要编号，不要解释。"}]},
    {"id": "arithmetic", "messages": [{"role": "user", "content": "商店有24支笔，卖出9支，又进货6支。现在有多少支？只回答数字。"}]},
    {"id": "ordering", "messages": [{"role": "user", "content": "将数字18、3、11、7按从小到大排列，只输出逗号分隔的数字。"}]},
    {"id": "rewrite", "messages": [{"role": "user", "content": "把“你必须马上把资料发给我”改写成一句礼貌的工作请求，不改变原意。"}]},
    {"id": "comparison", "messages": [{"role": "user", "content": "用两句话说明纸质笔记本和电子笔记各自的一个优点。"}]},
    {"id": "multi_turn", "messages": [{"role": "user", "content": "请记住：项目代号是青竹，负责人是小周。"}, {"role": "assistant", "content": "好的，项目代号是青竹，负责人是小周。"}, {"role": "user", "content": "负责人是谁？只回答姓名。"}]},
    {"id": "brief_explanation", "messages": [{"role": "user", "content": "用不超过三句话向小学生解释为什么下雨后有时会出现彩虹。"}]},
]

# Original diagnostic prose, not a benchmark or a claim of general model quality.
CORPUS = """清晨，社区图书馆打开大门，管理员先检查还书箱，再把新归还的书籍按类别放回书架。靠窗的阅览区比较安静，常有人在那里读报和做笔记。图书馆每周公布新书目录，也会根据读者的建议补充资料。借阅规则写在入口旁，读者可以查询借期和续借方式。工作人员希望大家保持桌面整洁，让下一位读者也能舒适地使用空间。

城市公交系统需要协调线路、班次和车辆维护。工作日的早晚高峰通常安排更多车辆，而客流较少的时段可以适当调整发车间隔。站牌提供方向和停靠站信息，乘客在上车前应确认终点站。遇到道路施工时，运营方会提前发布绕行通知。可靠的公共交通不仅依赖车辆数量，也依赖清晰的信息和持续的维护。

一个家庭准备做周末晚餐时，可以先查看冰箱里已有的食材，再决定需要购买什么。把菜单和购物清单一起写下来，有助于减少重复采购。处理生肉和蔬菜时应使用不同的工具，并及时清洁操作台。食物煮熟后如果不能马上吃完，应按照适当的条件保存。简单的计划能够节约时间，也让厨房工作更加有条理。

学校组织科学观察活动，让学生记录植物在不同光照条件下的生长情况。每组使用相同大小的花盆和相同种类的种子，并按统一的方法浇水。学生每天测量高度，记录叶片数量，还要说明观察时的天气。几周以后，大家整理数据并比较变化。实验结果不符合预期时，应先检查记录和条件，而不是立即删除不喜欢的数据。

软件项目的维护者接到错误报告后，首先确认用户遇到问题时的操作步骤。一个好的报告包含输入、期望结果、实际结果和软件版本。维护者尝试复现问题，再查找相关代码和日志。修复完成后，需要测试原先失败的场景，也要检查相邻功能是否受到影响。记录修复的原因，可以帮助后来接手的人理解设计选择。

河流经过山地、平原和城镇，把不同地区连接起来。沿岸植被能够稳定土壤，为许多动物提供栖息地。人们使用河水灌溉、运输和发电，同时也需要保护水质。判断一段河流的状况，不能只看某一天的颜色，还要结合长期监测和降雨信息。治理工作通常需要上游与下游共同协作，持续观察措施的效果。

制作一份清楚的会议纪要，重点是记录决定和后续行动。纪要不必逐字重现每个人的发言，但应保留重要的背景和不同意见。每项行动最好明确负责人和完成时间。如果讨论中有尚未解决的问题，也应单独列出。会后及时分享纪要，有助于参与者核对理解，并让没有到场的人了解工作进展。

学习一种新的技能需要练习，也需要适当的反馈。初学者可以把较大的目标拆成小步骤，每次集中改善其中一项。练习之后记录遇到的困难，比只记录花了多少时间更有帮助。休息同样重要，因为疲劳时容易反复犯同样的错误。随着经验增加，学习者能够调整方法，在速度和准确性之间找到合适的平衡。

地图用符号和比例表示现实世界，但任何地图都省略了一部分细节。道路地图突出路线和距离，地形图则更关注高度变化。使用地图时，要先阅读图例并确认方向。电子地图可以快速更新，也可能受到定位误差和网络条件的影响。出行前了解目的地的大致位置，能在设备暂时不可用时提供帮助。

小型商店管理库存时，需要记录商品的进货数量、销售数量和剩余数量。盘点可以发现记录与实物之间的差异，但差异不一定意味着商品丢失，也可能来自录入错误。经营者通常根据销售趋势决定下一次进货量。过多库存占用资金，过少库存又可能影响顾客购买。合理的库存安排应同时考虑需求、交货时间和保质期。

夜晚的观星活动适合在远离强光的地方进行。观察者可以先辨认明显的星座，再逐步寻找较暗的目标。月相、云量和空气透明度都会影响观察效果。望远镜能够放大细节，但初学者也可以用肉眼了解星空的基本结构。记录日期、时间和观察方向，有助于比较不同季节看到的天空。

设计一个公共花园时，需要考虑日照、水源、通行和维护。不同植物对环境的要求不同，不能只按花朵的颜色进行搭配。步道应方便行走，也要为轮椅和儿童推车预留空间。花园建成后，浇水、修剪和清理工作仍然需要长期安排。好的设计会让使用者感到舒适，也让维护人员能够持续照顾植物。

档案管理的目的，是让资料在需要时能够被可靠地找到。文件名称应该描述内容，并避免使用无法理解的缩写。版本发生变化时，应说明修改时间和主要内容。重要资料需要备份，但备份是否有效仍需定期检查。把文件复制到另一个地方，并不自动意味着它已经完整保存；恢复演练可以验证实际效果。

团队制定项目计划时，可以先列出必须完成的成果，再判断各项工作的依赖关系。有些任务能够同时进行，有些任务必须等待前一步提供结果。估算时间时，要为检查和修复留出空间。遇到新的信息，应及时调整计划并说明原因。计划的价值不在于永远不变，而在于帮助大家做出一致的行动。

观察气温变化时，测量位置和时间会影响结果。放在阳光直射处的温度计，与放在通风阴影处的温度计可能显示不同的数值。比较连续多天的数据，最好使用相同的仪器和方法。单次测量异常时，可以检查设备和周围环境。解释趋势需要足够的记录，不能仅凭一个数值推断长期变化。

编写使用说明时，作者应从读者实际要完成的任务出发。先说明操作目标，再按顺序列出必要步骤。涉及按钮和菜单时，名称应与界面保持一致。容易出错的地方可以提供简短的例子，但不应让例子掩盖主要流程。完成说明后，让没有参与编写的人试着操作，往往能发现遗漏的信息。

山间步道会随着天气和季节改变。干燥时容易通行的路段，在降雨后可能变得泥泞。出发前查看路线、预计时间和天气，有助于准备合适的装备。行进中应留意路标，并根据体力及时调整节奏。保护自然环境意味着带走垃圾、尊重野生动物，并沿已有路径行走，减少对植被的破坏。

衡量设备性能时，测试条件需要保持一致。冷启动和预热后的运行可能有明显差别，短时测试也未必反映持续工作的表现。记录输入规模、软件版本、温度和电源条件，能帮助解释结果。重复测试后报告中位数和变化范围，比只挑选最快的一次更有参考价值。性能指标应服务于实际使用需求。

博物馆展览把物品、文字和空间组织在一起，帮助参观者理解一个主题。一件展品的意义往往来自它的年代、用途和发现背景。说明文字需要准确，也要照顾不同年龄和知识水平的读者。互动装置可以增加参与感，但必须定期维护。策展团队会根据参观反馈改进路线，让信息更容易被理解。

整理个人预算时，可以先区分固定支出和可调整支出，再观察过去几个月的变化。记录的目的不是让每一笔消费都受到责备，而是帮助理解资源如何被使用。为不常发生的开支预留空间，可以减少临时压力。预算应随着家庭情况变化而调整，实际执行中的反馈比一开始的完美表格更重要。

社区举办志愿活动前，需要确定任务、时间和集合地点。报名者应了解工作内容和需要携带的物品。组织者安排分工时，应考虑参与者的经验和身体条件。活动结束后，清点工具并整理现场也是工作的一部分。简短的复盘能够总结有效的方法，让下一次合作更加顺畅。

录制声音时，房间的回声、背景噪声和麦克风距离都会影响效果。调整设备前，可以先录一小段样本并试听。讲话者保持适当的距离和稳定的音量，有助于获得清晰的录音。后期处理能够改善部分问题，但无法完全恢复已经丢失的信息。保存原始文件，可以在需要时重新进行编辑。

数据表格中的每一列应有清楚的含义。日期、数量和单位使用一致的格式，能够减少计算错误。缺失值与数值零并不相同，录入时应明确区分。对数据进行排序或筛选后，需要确认相关行仍然对应。分析结论应说明使用的数据范围和限制，避免让读者误以为表格包含所有情况。

修理自行车前，可以先观察故障发生的位置和条件。轮胎气压不足、链条润滑不良和刹车调整不当，会产生不同的骑行感受。工具使用后应放回固定位置，拆下的小零件也要妥善保存。修理完成后，先在安全区域进行短距离检查。对于无法确认的故障，及时寻求有经验的人帮助，通常比反复尝试更有效。
"""
# Two fixed copies ensure four complete windows even with a compact tokenizer.
# Only the first 2048 tokens are consumed; corpus hash and actual counts are reported.
CORPUS = CORPUS + "\n" + CORPUS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def settings() -> dict:
    return {"pipeline_version": 1, "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION, "llama_commit": LLAMA_COMMIT,
            "source_precision": "bf16", "quantization": "Q4_K_M",
            "imatrix": None, "seed": 42, "temperature": 0, "max_new_tokens": 128,
            "ppl_context": 512, "ppl_chunks": 4,
            "corpus_sha256": hashlib.sha256(CORPUS.encode()).hexdigest(),
            "prompts_sha256": hashlib.sha256(json.dumps(PROMPTS, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}


def fingerprint() -> str:
    return hashlib.sha256(json.dumps(settings(), sort_keys=True).encode()).hexdigest()


def cache_dir() -> Path:
    return DRIVE / "qwen3-rk3588-cache" / fingerprint()[:24]


def save_state() -> None:
    write_json(WORK / "run_state.json", STATE)


def available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable; cannot validate runtime memory")


def check_environment(drive: Path, free_disk: int, free_memory: int) -> None:
    if not drive.is_dir():
        raise RuntimeError("Google Drive is not mounted. Mount it interactively in Colab, then rerun; automatic mounting is disabled.")
    if free_disk < 15 * 2**30:
        raise RuntimeError("At least 15 GiB local free disk is required")
    if free_memory < 6 * 2**30:
        raise RuntimeError("At least 6 GiB MemAvailable is required")


def run(argv: list, label: str, timeout: int = 3600, env: dict | None = None) -> tuple[str, str]:
    LOGS.mkdir(parents=True, exist_ok=True)
    argv = [str(x) for x in argv]
    print(f"[{label}] {json.dumps(argv, ensure_ascii=False)}", flush=True)
    outpath, errpath = LOGS / f"{label}.stdout.log", LOGS / f"{label}.stderr.log"
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = str(SOURCE / "gguf-py")
    child_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if env:
        child_env.update(env)
    started = time.time()
    with outpath.open("w") as out, errpath.open("w") as err:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                env=child_env, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    stdout, stderr = outpath.read_text(errors="replace"), errpath.read_text(errors="replace")
    STATE.setdefault("commands", []).append({"label": label, "argv": argv, "returncode": code,
                                             "elapsed_s": round(time.time() - started, 3)})
    save_state()
    if code:
        raise RuntimeError(f"{label} failed ({code}); see {errpath}\n{stderr[-4000:]}\n{stdout[-1000:]}")
    print(f"[{label}] complete in {time.time()-started:.1f}s", flush=True)
    return stdout, stderr


def cache_valid(folder: Path, expected_fingerprint: str | None = None) -> tuple[bool, str]:
    try:
        manifest_path = folder / "manifest.json"
        marker = read_json(folder / "COMPLETE.json")
        if marker.get("manifest_sha256") != sha256(manifest_path):
            return False, "completion marker mismatch"
        manifest = read_json(manifest_path)
        if manifest.get("settings_fingerprint") != (expected_fingerprint or fingerprint()):
            return False, "settings/version mismatch"
        if manifest.get("status") != "validated":
            return False, "validation incomplete"
        model = manifest["model"]
        if model.get("filename") != MODEL_NAME:
            return False, "unexpected model filename"
        path = folder / MODEL_NAME
        if path.stat().st_size != model["size_bytes"] or sha256(path) != model["sha256"]:
            return False, "model hash/size mismatch"
        quality = manifest["quality_report"]
        if quality.get("filename") != "quality_report.json":
            return False, "unexpected report filename"
        if sha256(folder / "quality_report.json") != quality["sha256"]:
            return False, "quality report hash mismatch"
        validate_quality_binding(read_json(folder / "quality_report.json"), path)
        smoke = manifest["smoke_prompt"]
        if smoke.get("filename") != "smoke_prompt.txt" or sha256(folder / "smoke_prompt.txt") != smoke["sha256"]:
            return False, "smoke prompt hash mismatch"
        expected_sums = {MODEL_NAME: model["sha256"], "quality_report.json": quality["sha256"],
                         "manifest.json": sha256(manifest_path)}
        actual_sums = {}
        for line in (folder / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            actual_sums[name.strip()] = digest
        if actual_sums != expected_sums:
            return False, "checksum list mismatch"
        return True, "validated cache"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False, "missing or invalid cache files"


def validate_quality_binding(quality: dict, model: Path) -> None:
    """A successful report is valid only for these exact settings and model bytes."""
    try:
        result = quality["models"]["Q4_K_M"]
        if (quality.get("status") != "validated" or quality.get("settings") != settings()
                or result["size_bytes"] != model.stat().st_size
                or result["sha256"] != sha256(model)):
            raise RuntimeError("Refusing stale or unbound quality report")
    except (KeyError, TypeError, OSError) as exc:
        raise RuntimeError("Refusing incomplete quality report/model") from exc


def stage1_config() -> None:
    """1. Fixed provenance and deterministic validation settings."""
    global STATE
    WORK.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    old = WORK / "run_state.json"
    STATE = read_json(old) if old.exists() else {}
    if STATE.get("settings_fingerprint") != fingerprint():
        STATE = {"settings_fingerprint": fingerprint(), "settings": settings(), "commands": []}
    STATE["cache_hit"] = False
    STATE["completed_stages"] = []
    STATE.pop("failed_stage", None)
    STATE.pop("last_error", None)
    STATE["status"] = "running"
    save_state()
    print(json.dumps(settings(), ensure_ascii=False, indent=2))


def stage2_environment() -> None:
    """2. Headless environment guard. No drive.mount or new runtime."""
    free_disk, memory = shutil.disk_usage(WORK).free, available_memory()
    check_environment(DRIVE, free_disk, memory)
    STATE["environment"] = {"python": sys.version, "cpu_count": os.cpu_count(),
                            "threads": THREADS, "free_disk_bytes": free_disk,
                            "available_memory_bytes": memory, "backend": "CPU"}
    save_state()
    print(json.dumps(STATE["environment"], indent=2))


def stage3_cache() -> None:
    """3. Reuse only a complete, matching, fully rehashed cache."""
    valid, reason = cache_valid(cache_dir())
    STATE["cache_hit"] = valid
    STATE["cache_check"] = reason
    if valid:
        for name in [MODEL_NAME, "quality_report.json", "manifest.json", "SHA256SUMS", "smoke_prompt.txt", "COMPLETE.json"]:
            shutil.copy2(cache_dir() / name, ARTIFACTS / name)
        if sha256(ARTIFACTS / MODEL_NAME) != read_json(ARTIFACTS / "manifest.json")["model"]["sha256"]:
            raise RuntimeError("Copied cache failed local hash verification")
    save_state()
    print(f"cache_hit={valid}; {reason}; {cache_dir()}")


def prepare_python_environment() -> list[str]:
    """Repair partial Debian/Colab venvs without depending on ensurepip."""
    # Reapply configuration even when an interpreter survived an earlier failed
    # EnvBuilder call: failed ensurepip can leave system-site-packages disabled.
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(PYTHON.parent.parent)
    probe = subprocess.run([str(PYTHON), "-m", "pip", "--version"],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    if probe.returncode == 0:
        return [str(PYTHON), "-m", "pip"]
    # Recent pip can manage a pip-less virtual environment from the host Python.
    command = [sys.executable, "-m", "pip", "--python", str(PYTHON)]
    probe = subprocess.run(command + ["--version"], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=30)
    if probe.returncode:
        raise RuntimeError("Neither inherited pip nor host pip --python can manage the project venv")
    return command


def stage4_tools() -> None:
    """4. Pinned native CPU tools and isolated conversion dependencies."""
    if not (SOURCE / ".git").is_dir():
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", SOURCE], "clone_llama")
        run(["git", "-C", SOURCE, "pull", "--ff-only"], "pull_llama")
    head, _ = run(["git", "-C", SOURCE, "rev-parse", "HEAD"], "llama_head")
    status, _ = run(["git", "-C", SOURCE, "status", "--porcelain"], "llama_status")
    if status.strip():
        raise RuntimeError("Refusing to build an edited llama.cpp checkout")
    if head.strip() != LLAMA_COMMIT:
        branch, _ = run(["git", "-C", SOURCE, "branch", "--show-current"], "llama_branch")
        if not branch.strip():
            raise RuntimeError("Unexpected detached llama.cpp revision; use a fresh work directory")
        run(["git", "-C", SOURCE, "pull", "--ff-only"], "pull_llama_before_pin")
        run(["git", "-C", SOURCE, "fetch", "--depth", "1", "origin", LLAMA_COMMIT], "fetch_llama_pin")
        run(["git", "-C", SOURCE, "checkout", "--detach", LLAMA_COMMIT], "pin_llama")
    pip_command = prepare_python_environment()
    run(pip_command + ["install", "-r", SOURCE / "requirements/requirements-convert_hf_to_gguf.txt", "huggingface_hub"], "python_dependencies")
    run(["cmake", "-S", SOURCE, "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release",
         "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=OFF", "-DGGML_CUDA=OFF",
         "-DGGML_VULKAN=OFF", "-DGGML_METAL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
         "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_BUILD_SERVER=ON",
         "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_UI=OFF"], "configure_native")
    run(["cmake", "--build", BUILD, "--parallel", str(min(THREADS, 2)), "--target",
         "llama-quantize", "llama-cli", "llama-completion", "llama-perplexity"], "build_native", timeout=7200)
    packages, _ = run(pip_command + ["freeze"], "python_versions")
    (ARTIFACTS / "python-requirements-resolved.txt").write_text(packages)
    STATE["python_packages"] = packages.splitlines()
    save_state()


def stage5_download() -> None:
    """5. Download official weights at the fixed revision; no third-party fallback."""
    if STATE.get("cache_hit"):
        print("Complete cache already contains validated official-derived model; download skipped.")
        return
    code = """import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
repo, revision, dest = sys.argv[1:]
snapshot_download(repo_id=repo, revision=revision, local_dir=dest,
 allow_patterns=['*.safetensors', '*.json', '*.model', 'merges.txt', 'vocab.txt'], max_workers=2)
info = HfApi().model_info(repo, revision=revision, files_metadata=True)
assert info.sha == revision, (info.sha, revision)
files=[]
for sibling in info.siblings:
 p=Path(dest)/sibling.rfilename
 if p.is_file():
  lfs=sibling.lfs
  files.append({'filename':sibling.rfilename,'expected_sha256':getattr(lfs,'sha256',None)})
print(json.dumps({'revision':info.sha,'files':files}))
"""
    out, _ = run([PYTHON, "-c", code, MODEL_REPO, MODEL_REVISION, HF_DIR], "download_official", timeout=7200)
    provenance = json.loads(out.strip().splitlines()[-1])
    for item in provenance["files"]:
        p = HF_DIR / item["filename"]
        item["size_bytes"], item["sha256"] = p.stat().st_size, sha256(p)
        if item["expected_sha256"] and item["sha256"] != item["expected_sha256"]:
            raise RuntimeError(f"Official source checksum mismatch: {item['filename']}")
    config = read_json(HF_DIR / "config.json")
    if config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 28 or not config.get("tie_word_embeddings"):
        raise RuntimeError("Official model configuration differs from the planned architecture")
    STATE["source_files"] = provenance
    save_state()


def inspect_gguf(path: Path, label: str) -> dict:
    code = """import json, sys, collections
from gguf import GGUFReader
r=GGUFReader(sys.argv[1])
def val(k):
 f=r.fields.get(k)
 return f.contents() if f is not None else None
types=collections.Counter(t.tensor_type.name for t in r.tensors)
meta={'architecture':val('general.architecture'),'block_count':val('qwen3.block_count'),
 'file_type':val('general.file_type'),'tensor_count':len(r.tensors),'tensor_types':dict(types),
 'has_chat_template':bool(val('tokenizer.chat_template')),
 'has_input_embedding':any(t.name=='token_embd.weight' for t in r.tensors),
 'has_output_weight':any(t.name=='output.weight' for t in r.tensors),
 'vocab_size':len(r.fields['tokenizer.ggml.tokens'].data)}
print(json.dumps(meta, default=lambda x:x.item()))
"""
    out, _ = run([PYTHON, "-c", code, path], label)
    meta = json.loads(out.strip().splitlines()[-1])
    if meta["architecture"] != "qwen3" or meta["block_count"] != 28 or meta["vocab_size"] != 151936 or not meta["has_chat_template"]:
        raise RuntimeError(f"Unexpected GGUF metadata: {meta}")
    return meta


def stage6_convert_quantize() -> None:
    """6. Convert BF16 and quantize once; checkpoint each complete output."""
    model = ARTIFACTS / MODEL_NAME
    if STATE.get("cache_hit"):
        STATE["q4_metadata"] = inspect_gguf(model, "inspect_cached_q4")
        save_state()
        return
    checkpoint_path = WORK / "conversion_checkpoint.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {}
    if checkpoint.get("settings_fingerprint") != fingerprint():
        checkpoint = {"settings_fingerprint": fingerprint()}
    for kind, destination in [("bf16", WORK / BF16_NAME), ("q4", model)]:
        old = checkpoint.get(kind, {})
        if destination.is_file() and old.get("size_bytes") == destination.stat().st_size and old.get("sha256") == sha256(destination):
            print(f"Reusing fully verified {kind} conversion checkpoint")
            continue
        partial = destination.with_name(destination.name + ".partial")
        if partial.exists():
            partial.unlink()
        if kind == "bf16":
            run([PYTHON, SOURCE / "convert_hf_to_gguf.py", HF_DIR, "--outtype", "bf16", "--outfile", partial], "convert_bf16", timeout=7200)
        else:
            run([BUILD / "bin/llama-quantize", WORK / BF16_NAME, partial, "Q4_K_M", str(THREADS)], "quantize_q4_k_m", timeout=7200)
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Missing {kind} output")
        metadata = inspect_gguf(partial, f"inspect_{kind}")
        if kind == "q4" and not metadata["tensor_types"].get("Q4_K"):
            raise RuntimeError("Quantized file contains no Q4_K tensors")
        partial.replace(destination)
        checkpoint[kind] = {"size_bytes": destination.stat().st_size, "sha256": sha256(destination), "metadata": metadata}
        write_json(checkpoint_path, checkpoint)
    STATE["conversion"] = checkpoint
    save_state()


def extract_response(stdout: str) -> str:
    answer = re.sub(r"\s*\[end of text\]\s*$", "", stdout).strip()
    if not answer:
        raise RuntimeError("Model returned an empty response")
    if "<|im_start|>" in answer or "<|im_end|>" in answer or "<think>" in answer:
        raise RuntimeError("Unexpected template/thinking token in non-thinking response")
    return answer


def parse_ppl(stdout: str, stderr: str) -> float:
    text = stdout + "\n" + stderr
    if not re.search(r"calculating perplexity over 4 chunks, n_ctx=512", text):
        raise RuntimeError("PPL did not process exactly four 512-token chunks")
    matches = re.findall(r"Final estimate:\s*PPL\s*=\s*([^\s,+]+)", text)
    if not matches:
        raise RuntimeError("Missing final PPL estimate")
    value = float(matches[-1])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Non-finite or invalid PPL: {value}")
    return value


def completion(model: Path, prompt: Path, label: str, max_tokens: int = 128) -> dict:
    started = time.time()
    out, err = run([BUILD / "bin/llama-completion", "-m", model, "-ngl", "0", "-c", "1024",
                    "-b", "128", "-ub", "32", "-t", str(THREADS), "--no-conversation",
                    "--no-display-prompt", "--simple-io", "--color", "off", "--no-escape",
                    "--binary-file", prompt, "-n", str(max_tokens), "--seed", "42", "--temp", "0"], label, timeout=1800)
    return {"output": extract_response(out), "elapsed_s": round(time.time() - started, 3),
            "native_stdout_log": f"logs/{label}.stdout.log", "native_stderr_log": f"logs/{label}.stderr.log"}


def prepare_validation_inputs() -> dict:
    inputs = WORK / "validation_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    write_json(inputs / "prompts.json", {"prompts": PROMPTS})
    (inputs / "corpus.txt").write_text(CORPUS, encoding="utf-8")
    code = """import json,sys
from pathlib import Path
from transformers import AutoTokenizer
model, folder=sys.argv[1:]
p=Path(folder)
t=AutoTokenizer.from_pretrained(model, local_files_only=True)
rows=json.loads((p/'prompts.json').read_text())['prompts']
for row in rows:
 text=t.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
 assert text.count('<|im_start|>assistant')==sum(m['role']=='assistant' for m in row['messages'])+1
 assert text.endswith('<think>\\n\\n</think>\\n\\n'), repr(text[-100:])
 (p/(row['id']+'.txt')).write_text(text, encoding='utf-8')
n=len(t.encode((p/'corpus.txt').read_text(), add_special_tokens=False))
assert n>=2048, ('diagnostic corpus too short',n)
print(json.dumps({'corpus_token_count':n,'input_tokens_used':2048,'scored_tokens':1020,
 'ppl_chunks':4,'ppl_context':512,'template':'official tokenizer, enable_thinking=False'}))
"""
    out, _ = run([PYTHON, "-c", code, HF_DIR, inputs], "render_validation_inputs")
    return json.loads(out.strip().splitlines()[-1])


def stage7_validate() -> None:
    """7. Paired functional outputs + small-corpus PPL, or cached smoke."""
    if STATE.get("cache_hit"):
        # Cache includes its rendered smoke prompt, avoiding a fresh HF download.
        smoke = ARTIFACTS / "smoke_prompt.txt"
        if not smoke.is_file():
            raise RuntimeError("Cached smoke prompt is missing; remove COMPLETE.json and rebuild cache")
        copied = WORK / "cached_smoke_prompt.txt"
        shutil.copy2(smoke, copied)
        result = completion(ARTIFACTS / MODEL_NAME, copied, "cached_smoke", max_tokens=32)
        STATE["cache_smoke"] = result
        save_state()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    input_info = prepare_validation_inputs()
    inputs = WORK / "validation_inputs"
    quality = {"status": "running", "settings": settings(), "input_info": input_info,
               "backend": "CPU", "models": {},
               "limitations": "Small original diagnostic corpus and functional examples; not a general benchmark or a claim of lossless quantization. Does not validate PanVK."}
    for name, model in [("BF16", WORK / BF16_NAME), ("Q4_K_M", ARTIFACTS / MODEL_NAME)]:
        rows = []
        for prompt in PROMPTS:
            result = completion(model, inputs / f"{prompt['id']}.txt", f"{name}_{prompt['id']}")
            rows.append({"id": prompt["id"], "messages": prompt["messages"], **result})
            print(f"{name} / {prompt['id']}: {result['output']}", flush=True)
        out, err = run([BUILD / "bin/llama-perplexity", "-m", model, "-ngl", "0", "-c", "512",
                        "-b", "512", "-ub", "128", "-t", str(THREADS), "--chunks", "4",
                        "--binary-file", inputs / "corpus.txt"], f"{name}_ppl", timeout=7200)
        quality["models"][name] = {"functional_examples": rows, "ppl": parse_ppl(out, err),
                                   "size_bytes": model.stat().st_size, "sha256": sha256(model)}
        write_json(ARTIFACTS / "quality_report.incomplete.json", quality)
    bf, q4 = quality["models"]["BF16"]["ppl"], quality["models"]["Q4_K_M"]["ppl"]
    quality.update(status="validated", ppl_delta=q4 - bf, ppl_relative_change=q4 / bf - 1)
    write_json(ARTIFACTS / "quality_report.json", quality)
    STATE["quality"] = {"bf16_ppl": bf, "q4_ppl": q4, "ppl_relative_change": q4 / bf - 1}
    save_state()
    print(json.dumps(STATE["quality"], indent=2))


def stage8_publish() -> None:
    """8. Publish only validated artifacts; verify readback before COMPLETE."""
    model = ARTIFACTS / MODEL_NAME
    report = ARTIFACTS / "quality_report.json"
    if not model.is_file() or not report.is_file() or read_json(report).get("status") != "validated":
        raise RuntimeError("Refusing to publish missing or unvalidated model/report")
    quality = read_json(report)
    validate_quality_binding(quality, model)
    if not STATE.get("cache_hit"):
        expected_bf16 = STATE.get("conversion", {}).get("bf16", {})
        actual_bf16 = quality.get("models", {}).get("BF16", {})
        if not expected_bf16.get("sha256") or expected_bf16["sha256"] != actual_bf16.get("sha256"):
            raise RuntimeError("Refusing quality report for a different BF16 conversion")
    if STATE.get("cache_hit"):
        manifest = read_json(ARTIFACTS / "manifest.json")
    else:
        smoke = ARTIFACTS / "smoke_prompt.txt"
        shutil.copyfile(WORK / "validation_inputs/brief_explanation.txt", smoke)
        manifest = {"schema_version": 1, "model_repo": MODEL_REPO,
                    "model_revision": MODEL_REVISION, "llama_commit": LLAMA_COMMIT,
                    "quantization": "Q4_K_M", "source_precision": "bf16",
                    "settings_fingerprint": fingerprint(), "settings": settings(),
                    "status": "validated", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": {"filename": MODEL_NAME, "size_bytes": model.stat().st_size, "sha256": sha256(model)},
                    "quality_report": {"filename": report.name, "sha256": sha256(report)},
                    "smoke_prompt": {"filename": smoke.name, "sha256": sha256(smoke)},
                    "source_files": STATE.get("source_files"), "environment": STATE.get("environment"),
                    "conversion": STATE.get("conversion"), "commands": STATE.get("commands", []),
                    "python_packages": STATE.get("python_packages", []),
                    "board_validation": "not performed; deployment gate requires explicit confirmation"}
        write_json(ARTIFACTS / "manifest.json", manifest)
    checksums = "".join(f"{sha256(ARTIFACTS / name)}  {name}\n" for name in [MODEL_NAME, "quality_report.json", "manifest.json"])
    (ARTIFACTS / "SHA256SUMS").write_text(checksums)
    destination = cache_dir()
    destination.mkdir(parents=True, exist_ok=True)
    if not STATE.get("cache_hit"):
        marker = destination / "COMPLETE.json"
        if marker.exists():
            marker.unlink()
        for name in [MODEL_NAME, "quality_report.json", "manifest.json", "SHA256SUMS", "smoke_prompt.txt", "python-requirements-resolved.txt"]:
            src = ARTIFACTS / name
            dst = destination / name
            partial = destination / (name + ".partial")
            shutil.copyfile(src, partial)
            if sha256(partial) != sha256(src):
                raise RuntimeError(f"Drive temporary upload hash mismatch: {name}")
            partial.replace(dst)
            if sha256(dst) != sha256(src):
                raise RuntimeError(f"Drive final readback hash mismatch: {name}")
        # Store bounded complete native logs, not session history or credentials.
        logs_zip = shutil.make_archive(str(ARTIFACTS / "preparation-logs"), "zip", root_dir=WORK, base_dir="logs")
        shutil.copyfile(logs_zip, destination / "preparation-logs.zip")
        if sha256(Path(logs_zip)) != sha256(destination / "preparation-logs.zip"):
            raise RuntimeError("Drive log archive readback failed")
        write_json(marker, {"manifest_sha256": sha256(destination / "manifest.json")})
    valid, reason = cache_valid(destination)
    if not valid:
        raise RuntimeError(f"Final cache verification failed: {reason}")
    shutil.copyfile(destination / "COMPLETE.json", ARTIFACTS / "COMPLETE.json")
    if read_json(ARTIFACTS / "COMPLETE.json")["manifest_sha256"] != sha256(ARTIFACTS / "manifest.json"):
        raise RuntimeError("Local completion marker does not match local manifest")
    STATE["published_cache"] = str(destination)
    STATE["status"] = "model_preparation_complete"
    save_state()
    print(json.dumps({"status": STATE["status"], "cache": str(destination), "model": manifest["model"],
                      "board": "Not contacted. Stop at deployment gate after server build."}, ensure_ascii=False, indent=2))


STAGES = [stage1_config, stage2_environment, stage3_cache, stage4_tools,
          stage5_download, stage6_convert_quantize, stage7_validate, stage8_publish]


def run_stage(number: int) -> None:
    """Execute an ordered stage and stop downstream work after any failure.

    Some notebook runners keep executing later cells after an exception and
    still exit zero. Durable success markers, not the runner's exit code, are
    therefore the source of truth. Rerunning the failed stage is allowed once
    its prerequisites are complete; rerunning an earlier stage invalidates all
    later success markers.
    """
    if number not in range(1, len(STAGES) + 1):
        raise ValueError("Stage number must be between 1 and 8")
    if number > 1:
        if STATE.get("settings_fingerprint") != fingerprint():
            raise RuntimeError("Run stage 1 first; current stage settings are not initialized")
        completed = set(STATE.get("completed_stages", []))
        missing = [n for n in range(1, number) if n not in completed]
        if missing:
            raise RuntimeError(f"Stage {number} blocked: prerequisite stages {missing} have not succeeded. Fix and rerun the earliest failed stage.")
        STATE["completed_stages"] = sorted(n for n in completed if n < number)
        STATE.pop("failed_stage", None)
        STATE.pop("last_error", None)
        STATE["status"] = "running"
        save_state()
    try:
        STAGES[number - 1]()
    except BaseException as exc:
        STATE["failed_stage"] = number
        STATE["status"] = "failed"
        STATE["last_error"] = f"{type(exc).__name__}: {exc}"
        try:
            save_state()
        except OSError:
            pass
        raise
    STATE["completed_stages"] = sorted(set(STATE.get("completed_stages", [])) | {number})
    STATE.pop("failed_stage", None)
    STATE.pop("last_error", None)
    if number == 8:
        STATE["status"] = "model_preparation_complete"
    save_state()
    print(f"Stage {number}/8 passed; completed={STATE['completed_stages']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-stage", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--end-stage", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    if args.start_stage > args.end_stage:
        parser.error("start-stage must not exceed end-stage")
    if args.start_stage > 1:
        STATE.update(read_json(WORK / "run_state.json"))
        if STATE.get("settings_fingerprint") != fingerprint():
            raise RuntimeError("Resume settings differ; restart at stage 1")
    for number in range(args.start_stage, args.end_stage + 1):
        run_stage(number)


if __name__ == "__main__":
    main()
