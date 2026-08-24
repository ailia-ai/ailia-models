"""
Detect the minimum ailia SDK version required to open ONNX models.

site-packages に ailia_1210, ailia_1610 のような形式で配置された
バージョン別の ailia SDK を使用して、フォルダ内の全ての onnx を
古いバージョンから順に Open し、Open できた最小バージョンを出力する。

Usage:
    python3 util/required_version.py --required [dir]
"""

import argparse
import glob
import os
import re
import site
import subprocess
import sys

TIMEOUT_SEC = 300


def get_site_packages_dirs():
    dirs = []
    try:
        dirs.extend(site.getsitepackages())
    except AttributeError:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        dirs.append(user_site)
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


def find_versioned_modules(include_trial=False):
    pattern = re.compile(r"^ailia_\d+(_\d+)?(_trial)?$")
    modules = []
    for sp in get_site_packages_dirs():
        for name in os.listdir(sp):
            if not pattern.match(name):
                continue
            if name.endswith("_trial") and not include_trial:
                continue
            if os.path.isfile(os.path.join(sp, name, "__init__.py")):
                modules.append(name)
    return sorted(set(modules))


def get_module_version(module):
    result = subprocess.run(
        [sys.executable, "-c",
         "import {}; print({}.get_version())".format(module, module)],
        capture_output=True, text=True, timeout=TIMEOUT_SEC)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def parse_version(version_string):
    # e.g. "1.2.10.686-r22439cc14" -> (1, 2, 10, 686)
    numbers = version_string.split("-")[0]
    return tuple(int(x) for x in numbers.split(".") if x.isdigit())


def try_open(module, onnx_path, verbose=False):
    code = ("import {m}; {m}.Net(None, r'''{p}''')"
            .format(m=module, p=onnx_path))
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return False
    if verbose and result.returncode != 0:
        message = result.stderr.strip().splitlines()
        if message:
            print("    {} : {}".format(module, message[-1]))
    return result.returncode == 0


def detect_required_versions(target_dir, verbose=False, include_trial=False):
    onnx_files = sorted(glob.glob(os.path.join(target_dir, "*.onnx")))
    if not onnx_files:
        print("No onnx files found in {}".format(os.path.abspath(target_dir)))
        print("Please run the model once without --required "
              "to download the onnx files.")
        return

    print("Collecting ailia SDK versions from site-packages...")
    versions = []
    for module in find_versioned_modules(include_trial):
        version_string = get_module_version(module)
        if version_string is None:
            print("  {} : failed to import, skipped".format(module))
            continue
        versions.append((parse_version(version_string), version_string, module))
    if not versions:
        print("No versioned ailia SDK (ailia_XXXX) found in site-packages")
        return
    versions.sort()
    print("Found {} versions: {}".format(
        len(versions), ", ".join(v[1].split("-")[0] for v in versions)))
    print()

    for onnx_path in onnx_files:
        print("{} :".format(os.path.basename(onnx_path)))
        required = None
        for _, version_string, module in versions:
            if try_open(module, onnx_path, verbose):
                required = (version_string, module)
                break
        if required:
            print("  minimum version = {} ({})".format(
                required[0].split("-")[0], required[1]))
        else:
            print("  could not be opened with any available version")


def main():
    parser = argparse.ArgumentParser(
        description="Detect the minimum ailia SDK version "
                    "required to open ONNX models.")
    parser.add_argument(
        "dir", nargs="?", default=".",
        help="target directory containing onnx files (default: current dir)")
    parser.add_argument(
        "--required", action="store_true",
        help="detect the minimum ailia SDK version for each onnx")
    parser.add_argument(
        "--verbose", action="store_true",
        help="show error messages of failed open")
    parser.add_argument(
        "--include_trial", action="store_true",
        help="include trial versions (ailia_XXXX_trial)")
    args = parser.parse_args()

    if not args.required:
        parser.print_help()
        sys.exit(1)

    detect_required_versions(args.dir, args.verbose, args.include_trial)


if __name__ == "__main__":
    main()
