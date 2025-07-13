import subprocess
import json
import os
import sys
import csv
import shutil

SKIP_ENVS = ["base", "dali_env"]  # Add any environments you want to protect

def log(title):
    print(f"\n{'='*40}\n{title}\n{'='*40}")

def get_conda_envs():
    conda_exe = shutil.which("conda")
    if not conda_exe:
        print("❌ Conda not found in PATH.")
        return []
    try:
        result = subprocess.run([conda_exe, "env", "list", "--json"], capture_output=True, text=True)
        envs = json.loads(result.stdout).get("envs", [])
        return envs
    except Exception as e:
        print("❌ Error getting Conda environments:", e)
        return []

def get_pip_packages(pip_cmd):
    try:
        result = subprocess.run([pip_cmd, "list", "--format", "json"], capture_output=True, text=True)
        return json.loads(result.stdout)
    except:
        return []

def scan_conda_envs():
    log("📦 Scanning Conda Environments")
    envs = get_conda_envs()
    all_env_packages = {}

    for env_path in envs:
        env_name = os.path.basename(env_path)
        # Skip invalid or protected environments
        if env_name.lower() in SKIP_ENVS or "miniconda" in env_name.lower() or not os.path.isdir(env_path):
            continue
        pip_cmd = os.path.join(env_path, "Scripts", "pip.exe") if sys.platform == "win32" else os.path.join(env_path, "bin", "pip")
        if not os.path.exists(pip_cmd):
            continue
        pkgs = get_pip_packages(pip_cmd)
        all_env_packages[env_name] = pkgs
        print(f"✅ {env_name}: {len(pkgs)} packages")
    return all_env_packages

def scan_global_pip():
    log("🌐 Scanning Global Pip Installation")
    pkgs = get_pip_packages("pip")
    print(f"✅ Global pip: {len(pkgs)} packages")
    return pkgs

def scan_system_paths():
    log("🔍 Detecting All Python Installations")
    py_paths = []

    if sys.platform == "win32":
        try:
            result = subprocess.run(["where", "python"], capture_output=True, text=True)
            paths = result.stdout.strip().split("\n")
            for p in paths:
                if os.path.exists(p.strip()):
                    py_paths.append(p.strip())
        except:
            pass
    else:
        try:
            result = subprocess.run(["which", "-a", "python"], capture_output=True, text=True)
            py_paths += result.stdout.strip().split("\n")
        except:
            pass

    print(f"✅ Found {len(py_paths)} Python executables")
    return py_paths

def remove_conda_env(env_name):
    print(f"🧹 Removing Conda environment: {env_name}")
    conda_exe = shutil.which("conda")
    if conda_exe:
        subprocess.run([conda_exe, "env", "remove", "-n", env_name])
    else:
        print("❌ Conda not found in PATH. Skipping environment removal.")

def clean_conda_cache():
    log("🧼 Cleaning Conda Cache")
    conda_exe = shutil.which("conda")
    if conda_exe:
        subprocess.run([conda_exe, "clean", "--all", "--yes"])
    else:
        print("❌ Conda not found in PATH. Skipping cache cleanup.")

def clean_pip_cache():
    log("🧹 Cleaning Pip Cache")
    subprocess.run(["pip", "cache", "purge"])

def export_to_csv(env_data, global_data, file_path="packages_audit.csv"):
    log("📤 Exporting to packages_audit.csv")
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Environment", "Package", "Version"])
        for env, pkgs in env_data.items():
            for pkg in pkgs:
                writer.writerow([env, pkg["name"], pkg["version"]])
        for pkg in global_data:
            writer.writerow(["Global", pkg["name"], pkg["version"]])
    print(f"✅ Saved report to {file_path}")

def audit():
    env_data = scan_conda_envs()
    global_data = scan_global_pip()
    system_paths = scan_system_paths()
    export_to_csv(env_data, global_data)

    log("🧠 Summary")
    print(f"🔧 Conda environments scanned: {len(env_data)}")
    print(f"🌍 Global pip packages: {len(global_data)}")
    print(f"🐍 Python installations found: {len(system_paths)}")

    choice = input("\n🚨 Clean unused Conda environments and pip cache? (y/n): ").lower()
    if choice == "y":
        for env_name in env_data.keys():
            if env_name not in SKIP_ENVS:
                remove_conda_env(env_name)
        clean_conda_cache()
        clean_pip_cache()
        print("\n✅ Cleanup complete.")
    else:
        print("\n❎ No changes made. Review packages_audit.csv for manual cleanup.")

if __name__ == "__main__":
    audit()
