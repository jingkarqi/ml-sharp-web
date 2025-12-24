from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
OUTPUT_DIR = REPO_ROOT / "output"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
MODEL_PATH = REPO_ROOT / "src" / "models" / "sharp_2572gikvuh.pt"
MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
OPTIONAL_DEPENDENCIES = {
    "hf-xet": "HuggingFace 下载加速（仅在需要下载模型时有用）",
    "pillow-heif": "支持 HEIC 图片（iPhone 照片）",
}


def print_section(title: str) -> None:
    print("")
    print(f"== {title} ==")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "是", "好", "ok"}:
            return True
        if answer in {"n", "no", "否"}:
            return False
        print("请输入 Y 或 N。")


def ensure_venv() -> Path:
    print_section("虚拟环境检查")
    if VENV_DIR.exists():
        print(f"已检测到虚拟环境: {VENV_DIR}")
    else:
        print("未检测到虚拟环境，正在创建 .venv ...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print("虚拟环境创建完成。")

    if sys.platform.startswith("win"):
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"

    if not venv_python.exists():
        raise FileNotFoundError(f"无法找到虚拟环境解释器: {venv_python}")
    return venv_python


def run_venv_python(venv_python: Path, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(venv_python), "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def ensure_pip(venv_python: Path) -> None:
    result = subprocess.run(
        [str(venv_python), "-m", "pip", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        if result.stdout.strip():
            print(result.stdout.strip())
        return
    subprocess.run([str(venv_python), "-m", "ensurepip", "--upgrade"], check=False)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(dest.suffix + ".part")
    print(f"开始下载模型: {url}")
    try:
        with urllib.request.urlopen(url) as response:
            total = response.headers.get("Content-Length")
            total_size = int(total) if total and total.isdigit() else 0
            downloaded = 0
            with temp_path.open("wb") as file_handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file_handle.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = downloaded / total_size * 100
                        print(
                            f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)",
                            end="",
                        )
        if total_size:
            print("")
        temp_path.replace(dest)
        print(f"模型已保存到: {dest}")
    except Exception as exc:  # noqa: BLE001
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        print(f"模型下载失败: {exc}")
        print("请检查网络连接，或手动下载后放到指定目录。")


def ensure_model() -> None:
    print_section("模型权重检查")
    if MODEL_PATH.exists():
        print(f"模型权重已存在: {MODEL_PATH}")
        return

    print("未发现模型权重文件。")
    if ask_yes_no("是否自动下载模型？", default=True):
        download_file(MODEL_URL, MODEL_PATH)
    else:
        print(f"请将模型权重放置到: {MODEL_PATH.parent}")


def marker_applies_simple(marker: str) -> bool:
    marker_lower = marker.lower()
    sys_plat = sys.platform.lower()
    machine = platform.machine().lower()

    if "sys_platform" in marker_lower:
        if "win32" in marker_lower:
            return sys_plat.startswith("win")
        if "linux" in marker_lower:
            return sys_plat.startswith("linux")
        if "darwin" in marker_lower:
            return sys_plat.startswith("darwin")

    if "platform_machine" in marker_lower:
        if "x86_64" in marker_lower or "amd64" in marker_lower:
            return machine in {"x86_64", "amd64"}
        if "aarch64" in marker_lower or "arm64" in marker_lower:
            return machine in {"aarch64", "arm64"}

    return True


def extract_name(req: str) -> str:
    base = req.strip()
    for sep in ["==", ">=", "<=", "~=", ">", "<"]:
        if sep in base:
            base = base.split(sep, 1)[0]
            break
    base = base.split("[", 1)[0]
    return base.strip()


def load_requirements() -> list[str]:
    if not REQUIREMENTS_PATH.is_file():
        return []
    lines: list[str] = []
    for raw in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def get_installed_packages(venv_python: Path) -> dict[str, str]:
    code = (
        "import json, importlib.metadata as m;"
        "pkgs = {d.metadata.get('Name', d.metadata.get('Summary', '')).lower(): d.version "
        "for d in m.distributions() if d.metadata.get('Name')};"
        "print(json.dumps(pkgs))"
    )
    result = run_venv_python(venv_python, code)
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {}


def check_non_torch_packages(venv_python: Path) -> list[str]:
    installed = get_installed_packages(venv_python)
    missing_required: list[str] = []
    missing_optional: list[str] = []
    missing_editable = False

    for entry in load_requirements():
        if entry.startswith("-e "):
            target = entry[3:].strip()
            if target == ".":
                if "sharp" not in installed:
                    missing_editable = True
            continue
        if entry.startswith("-r ") or entry.startswith("--"):
            continue

        marker = ""
        if ";" in entry:
            entry, marker = entry.split(";", 1)
            entry = entry.strip()
            marker = marker.strip()
            if marker and not marker_applies_simple(marker):
                continue

        name = extract_name(entry)
        if not name:
            continue

        if name.lower().startswith("torch"):
            continue
        if name.lower() in {"torchvision", "torchaudio"}:
            continue

        if name.lower() not in installed:
            if name.lower() in OPTIONAL_DEPENDENCIES:
                missing_optional.append(entry.strip())
            else:
                missing_required.append(entry.strip())

    return missing_required, missing_optional, missing_editable


def install_packages(venv_python: Path, specs: list[str], title: str) -> bool:
    if not specs:
        return True
    print(title)
    for spec in specs:
        print(f"  - {spec}")
    if not ask_yes_no("是否继续安装？", default=True):
        return False
    cmd = [str(venv_python), "-m", "pip", "install", *specs]
    success = subprocess.run(cmd, check=False).returncode == 0
    if not success:
        print("依赖安装失败，请检查网络或手动执行安装。")
    return success


def select_inference_mode() -> str:
    print_section("推理方式选择")
    print("请选择推理方式：")
    print("  1) CPU")
    print("  2) GPU (CUDA)")
    while True:
        choice = input("请输入 1 或 2: ").strip()
        if choice == "1":
            return "cpu"
        if choice == "2":
            return "cuda"
        print("输入无效，请重新输入。")


def get_torch_info(venv_python: Path) -> dict[str, str | bool | None]:
    code = (
        "import json\n"
        "info = {}\n"
        "try:\n"
        "    import torch\n"
        "    info['available'] = True\n"
        "    info['version'] = torch.__version__\n"
        "    info['cuda_version'] = torch.version.cuda\n"
        "    info['cuda_available'] = torch.cuda.is_available()\n"
        "except Exception as exc:\n"
        "    info['available'] = False\n"
        "    info['error'] = str(exc)\n"
        "print(json.dumps(info))\n"
    )
    result = run_venv_python(venv_python, code)
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"available": False, "error": "无法解析 torch 信息"}


def has_nvidia_smi() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def get_nvidia_cuda_version() -> tuple[int, int] | None:
    try:
        result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", result.stdout)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def suggest_cuda_tag() -> tuple[str, str]:
    env_tag = os.getenv("SHARP_CUDA_TAG", "").strip()
    if env_tag:
        return env_tag, "来自环境变量 SHARP_CUDA_TAG"

    cuda_version = get_nvidia_cuda_version()
    if cuda_version:
        major, minor = cuda_version
        if major >= 13:
            return f"cu{major}{minor}", f"检测到 CUDA {major}.{minor}"
        if major == 12:
            if minor >= 9:
                return "cu129", "检测到 CUDA 12.9"
            if minor >= 8:
                return "cu128", "检测到 CUDA 12.8"
            if minor >= 6:
                return "cu126", "检测到 CUDA 12.6+"
        if major == 11:
            if minor >= 8:
                return "cu118", "检测到 CUDA 11.8"

    return "cu128", "无法判断驱动版本，默认选择 cu128"


def install_torch(venv_python: Path, mode: str) -> None:
    print_section("Torch 安装")
    torch_info = get_torch_info(venv_python)
    torch_available = torch_info.get("available", False)
    cuda_version = torch_info.get("cuda_version")

    if mode == "cuda":
        if torch_available and cuda_version:
            print(f"已检测到 CUDA 版 Torch: {torch_info.get('version')}")
            return
        if not has_nvidia_smi():
            print("未检测到 nvidia-smi，可能未安装显卡驱动或无 NVIDIA 显卡。")
            if not ask_yes_no("仍然继续安装 CUDA 版 Torch 吗？", default=False):
                return
        print("未检测到 CUDA 版 Torch，将进行安装或升级。")
        default_tag, reason = suggest_cuda_tag()
        prompt = (
            f"请输入 CUDA 轮子标签（如 cu129/cu128/cu126/cu130），"
            f"回车默认 {default_tag}（{reason}）: "
        )
        cuda_tag = input(prompt).strip()
        if not cuda_tag:
            cuda_tag = default_tag
        cmd = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--upgrade",
            "--index-url",
            f"https://download.pytorch.org/whl/{cuda_tag}",
        ]
    else:
        if torch_available:
            print(f"已检测到 Torch: {torch_info.get('version')}")
            return
        print("未检测到 Torch，将安装 CPU 版。")
        cmd = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--upgrade",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]

    subprocess.run(cmd, check=False)


def main() -> int:
    print("SHARP 初始化脚本（中文引导）")
    venv_python = ensure_venv()
    ensure_pip(venv_python)

    ensure_model()

    print_section("依赖检查（除 Torch 外）")
    missing_required, missing_optional, missing_editable = check_non_torch_packages(venv_python)
    if not missing_required and not missing_optional and not missing_editable:
        print("依赖齐全，无需安装。")
    else:
        if missing_required:
            install_packages(venv_python, missing_required, "检测到缺失的核心依赖:")
        if missing_optional:
            print("检测到缺失的可选依赖:")
            for spec in missing_optional:
                name = extract_name(spec).lower()
                desc = OPTIONAL_DEPENDENCIES.get(name, "")
                if desc:
                    print(f"  - {spec}（{desc}）")
                else:
                    print(f"  - {spec}")
            if ask_yes_no("是否安装可选依赖？", default=False):
                install_packages(venv_python, missing_optional, "开始安装可选依赖:")
        if missing_editable:
            print("未检测到项目本体（sharp）。")
            if ask_yes_no("是否尝试执行 pip install -e . 以启用 sharp 命令？", default=False):
                result = subprocess.run(
                    [str(venv_python), "-m", "pip", "install", "-e", "."],
                    check=False,
                )
                if result.returncode != 0:
                    print("安装失败。你仍然可以使用 PYTHONPATH=src 来运行 CLI。")
            else:
                print("跳过安装项目本体。若需 CLI，请使用 PYTHONPATH=src。")

    mode = select_inference_mode()
    install_torch(venv_python, mode)

    print_section("完成")
    print("初始化流程完成。")

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
