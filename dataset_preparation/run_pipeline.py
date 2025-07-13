import subprocess, time, logging, sys, yaml
from pathlib import Path

# List of required step scripts
REQUIRED_SCRIPTS = [
    "convert_to_yolo_format.py",
    "generate_masks.py",
    "extract_graph_relations.py",
    "generate_action_programs.py",
    "split_dataset.py",
    "normalize_and_sort.py"
]

# Initialize logging
dataset_root = Path(__file__).parent.resolve()
log_dir = dataset_root / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Check for missing scripts
missing_scripts = [s for s in REQUIRED_SCRIPTS if not (dataset_root / s).exists()]
if missing_scripts:
    logging.error(f"Missing required pipeline scripts: {', '.join(missing_scripts)}")
def step_convert_to_yolo():
    subprocess.run([
        sys.executable, "convert_to_yolo_format.py"], check=True)
            sys.executable, "augment.py", "--config", str(dataset_root / "config.yaml")
def step_generate_masks():
    subprocess.run([
        sys.executable, "generate_masks.py"], check=True)
        ], check=True)
def step_extract_graph_relations():
    subprocess.run([
        sys.executable, "extract_graph_relations.py"], check=True)

def step_generate_action_programs():
    subprocess.run([
        sys.executable, "generate_action_programs.py"], check=True)
def step_auto_label():
def step_split_dataset():
    subprocess.run([
        sys.executable, "split_dataset.py"], check=True)
    if cfg["pipeline"].get("run_auto_label", False):
def step_normalize_and_sort():
    subprocess.run([
        sys.executable, "normalize_and_sort.py"], check=True)
        subprocess.run([
            sys.executable, "auto_label.py", "--config", str(dataset_root / "config.yaml")
def main():
    timed_step("Convert to YOLO Format", step_convert_to_yolo)
    timed_step("Generate Masks", step_generate_masks)
    timed_step("Extract Graph Relations", step_extract_graph_relations)
    timed_step("Generate Action Programs", step_generate_action_programs)
    timed_step("Split Dataset", step_split_dataset)
    timed_step("Normalize and Sort Labels", step_normalize_and_sort)
    logging.info("🎉 All dataset preparation steps completed successfully.")
        ], check=True)

def step_synthesis():
    if cfg["pipeline"].get("run_synthesis", False):
        subprocess.run([
            sys.executable, "synthesize.py", "--config", str(dataset_root / "config.yaml")
        ], check=True)

def step_metadata():
    subprocess.run([
        sys.executable, "metadata_logger.py", "--config", str(dataset_root / "config.yaml")
    ], check=True)

def step_curriculum():
    if cfg["pipeline"].get("run_curriculum", False):
        subprocess.run([
            sys.executable, "curriculum_sampler.py", "--config", str(dataset_root / "config.yaml")
        ], check=True)

def main():
    timed_step("Prepare Dataset", step_prepare)
    timed_step("Augmentation",      step_augment)
    timed_step("Auto-Labeling",     step_auto_label)
    timed_step("Synthesis",         step_synthesis)
    timed_step("Metadata Logging",  step_metadata)
    timed_step("Curriculum Split",  step_curriculum)
    logging.info("🎉 All dataset preparation steps completed successfully.")

if __name__ == "__main__":
    main()
