"""Build only the standalone capture runtime and Qwen3.6 capture kernels."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sdk", type=Path, required=True, help="XRT SDK root")
    ap.add_argument("--iron", type=Path, help="IRON environment root, required for kernel build")
    ap.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.6-35B-A3B-NPU2")
    ap.add_argument("--spec", type=Path, help="explicit kernel ModelSpec (otherwise derive from the complete model)")
    ap.add_argument("--kernels", type=Path, default=ROOT / "src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels_capture")
    ap.add_argument("--only", choices=["runtime", "kernels", "both"], default="both")
    a = ap.parse_args()
    sdk = a.sdk.resolve()
    if a.only in ("runtime", "both"):
        source = ROOT / "src/open_qwen36"
        build = source / "out/build"
        cmd = ["cmake", "-S", str(source), "-B", str(build),
               f"-DXRT_INCLUDE_DIR={sdk / 'include'}", f"-DXRT_LIB_DIR={sdk / 'lib'}"]
        subprocess.run(cmd, check=True)
        subprocess.run(["cmake", "--build", str(build), "--config", "Release",
                        "--target", "open_qwen36_cli", "--parallel", "4"], check=True)
        exe = "open_qwen36_cli.exe" if os.name == "nt" else "open_qwen36_cli"
        built = build / "Release" / exe if (build / "Release" / exe).is_file() else build / exe
        shutil.copy2(built, source / "out" / exe)
    if a.only in ("kernels", "both"):
        if not a.iron:
            ap.error("--iron is required for kernel build")
        if not a.spec and not (a.model / "model.q4nx").is_file():
            ap.error("text weights are required to derive their native quantization; alternatively supply --spec")
        iron = a.iron.resolve()
        python = iron / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        site = iron / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
        env = os.environ.copy()
        env.update(MLIR_AIE_INSTALL_DIR=str(site / "mlir_aie"), PEANO_INSTALL_DIR=str(site / "llvm-aie"),
                   XRT_ROOT=str(sdk), NPU2="1", PYTHONUTF8="1")
        env["PATH"] = os.pathsep.join(map(str, [sdk, sdk / "lib", site / "mlir_aie/bin", site / "mlir_aie/lib"])) + os.pathsep + env["PATH"]
        env["PYTHONPATH"] = str(sdk / "python") + os.pathsep + env.get("PYTHONPATH", "")
        spec_args = ["--spec", str(a.spec.resolve())] if a.spec else ["--model-dir", str(a.model.resolve())]
        subprocess.run([str(python), str(ROOT / "open_kernels/export_qwen36_kernels.py"),
                        "--capture", *spec_args, "--out", str(a.kernels.resolve())],
                       cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
