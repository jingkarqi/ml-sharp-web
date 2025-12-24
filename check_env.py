from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "src" / "models" / "sharp_2572gikvuh.pt"


def format_bytes(num_bytes: int | float | None) -> str:
    if not num_bytes:
        return "0 B"
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    precision = 0 if size >= 10 or unit_index == 0 else 1
    return f"{size:.{precision}f} {units[unit_index]}"


def in_virtual_env() -> bool:
    if os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"):
        return True
    return hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix


def get_memory_total() -> str:
    try:
        import psutil  # type: ignore

        return format_bytes(psutil.virtual_memory().total)
    except Exception:
        return "unknown"


def get_torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch  # type: ignore

        info["available"] = True
        info["version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["device_name"] = torch.cuda.get_device_name(0)
        else:
            info["device_name"] = None
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def get_nvidia_smi() -> list[str] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines or None


def resolve_checkpoint() -> Path | None:
    requested = os.getenv("SHARP_CHECKPOINT", "").strip()
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.is_file():
            return candidate
        return None
    if DEFAULT_CHECKPOINT_PATH.is_file():
        return DEFAULT_CHECKPOINT_PATH
    return None


def load_requirements() -> list[str]:
    req_path = REPO_ROOT / "requirements.txt"
    if not req_path.is_file():
        return []
    requirements: list[str] = []
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirements.append(line)
    return requirements


def parse_requirements(lines: list[str]) -> list[Any]:
    parsed: list[Any] = []
    for line in lines:
        if line.startswith("-e "):
            parsed.append({"editable": line[3:].strip()})
            continue
        if line.startswith("-r "):
            continue
        if line.startswith("--"):
            continue
        parsed.append(line)
    return parsed


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


def check_packages() -> dict[str, list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []
    ok: list[str] = []

    requirements = parse_requirements(load_requirements())

    try:
        from packaging.requirements import Requirement  # type: ignore
    except Exception:
        Requirement = None  # type: ignore

    for entry in requirements:
        if isinstance(entry, dict) and "editable" in entry:
            try:
                metadata.version("sharp")
                ok.append("sharp（可编辑）")
            except metadata.PackageNotFoundError:
                missing.append("sharp（可编辑）")
            continue

        if Requirement is not None:
            try:
                req = Requirement(entry)
            except Exception:
                req = None
        else:
            req = None

        if req is not None:
            if req.marker and not req.marker.evaluate():
                continue
            name = req.name
            spec = req.specifier
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                missing.append(name)
                continue
            if spec and version not in spec:
                mismatched.append(f"{name}（当前 {version}，期望 {spec}）")
            else:
                ok.append(f"{name}（{version}）")
        else:
            name = entry.split(";", 1)[0].strip()
            marker = entry.split(";", 1)[1].strip() if ";" in entry else ""
            if marker and not marker_applies_simple(marker):
                continue
            if not name:
                continue
            base = name.split("==", 1)[0].strip()
            try:
                version = metadata.version(base)
                ok.append(f"{base}（{version}）")
            except metadata.PackageNotFoundError:
                missing.append(base)

    return {"missing": missing, "mismatched": mismatched, "ok": ok}


def build_report() -> str:
    report: list[str] = []
    report.append("SHARP 环境检查报告")
    report.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("Python")
    report.append(f"  版本: {platform.python_version()}")
    report.append(f"  解释器: {sys.executable}")
    report.append(f"  虚拟环境: {'已激活' if in_virtual_env() else '未激活'}")
    report.append(f"  .venv 目录: {'存在' if (REPO_ROOT / '.venv').exists() else '缺失'}")
    report.append("")

    report.append("硬件")
    report.append(f"  操作系统: {platform.system()} {platform.release()}")
    report.append(f"  架构: {platform.machine()}")
    report.append(f"  CPU 核心数: {os.cpu_count() or 'unknown'}")
    report.append(f"  内存: {get_memory_total()}")

    torch_info = get_torch_info()
    if torch_info.get("available"):
        report.append(f"  Torch: {torch_info.get('version')}")
        report.append(f"  Torch CUDA: {torch_info.get('cuda_version')}")
        report.append(f"  CUDA 可用: {torch_info.get('cuda_available')}")
        report.append(f"  CUDA 设备数量: {torch_info.get('device_count')}")
        if torch_info.get("device_name"):
            report.append(f"  CUDA 设备: {torch_info.get('device_name')}")
    else:
        report.append("  Torch: 无法导入")
        if torch_info.get("error"):
            report.append(f"  Torch 错误: {torch_info.get('error')}")

    smi = get_nvidia_smi()
    if smi:
        report.append("  nvidia-smi: 已检测到")
        for line in smi:
            report.append(f"    {line}")
    else:
        report.append("  nvidia-smi: 不可用")
    report.append("")

    report.append("模型检查点")
    checkpoint = resolve_checkpoint()
    if checkpoint:
        report.append(f"  状态: 已找到 ({checkpoint})")
    else:
        report.append("  状态: 缺失")
        report.append(f"  预期位置: {DEFAULT_CHECKPOINT_PATH}")
        report.append("  提示: 设置 SHARP_CHECKPOINT 或启用 SHARP_ALLOW_DOWNLOAD=1")
    report.append("")

    report.append("依赖包")
    pkg_status = check_packages()
    report.append(f"  缺失: {len(pkg_status['missing'])}")
    for name in pkg_status["missing"]:
        report.append(f"    - {name}")
    report.append(f"  版本不匹配: {len(pkg_status['mismatched'])}")
    for name in pkg_status["mismatched"]:
        report.append(f"    - {name}")
    report.append(f"  正常: {len(pkg_status['ok'])}")
    report.append("")

    return "\n".join(report)


def main() -> int:
    report = build_report()
    print(report)

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = OUTPUT_DIR / "env_report.txt"
        report_path.write_text(report + "\n", encoding="utf-8")
        print("")
        print(f"报告已保存到: {report_path}")
    except Exception as exc:  # noqa: BLE001
        print("")
        print(f"保存报告失败: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
