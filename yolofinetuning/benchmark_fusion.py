# benchmark_fusion.py
import time
import numpy as np
import json
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Import CPU and GPU fusion modules
try:
    from symbolic_fusion_cpu import fuse_masks_cpu, build_relation_graph_cpu
    logger.info("Successfully imported CPU symbolic fusion modules.")
except ImportError as e:
    logger.error(f"Failed to import symbolic_fusion_cpu: {e}. CPU benchmark will be skipped.")
    fuse_masks_cpu = None
    build_relation_graph_cpu = None

try:
    from symbolic_fusion_gpu import fuse_masks_gpu, build_relation_graph_gpu
    # Check if CuPy is available, as symbolic_fusion_gpu depends on it
    import cupy as cp
    _ = cp.array([1]) # Simple test to ensure CuPy is functional
    logger.info("Successfully imported GPU symbolic fusion modules and CuPy is functional.")
except ImportError as e:
    logger.error(f"Failed to import symbolic_fusion_gpu or CuPy is not available/functional: {e}. GPU benchmark will be skipped.")
    fuse_masks_gpu = None
    build_relation_graph_gpu = None
except Exception as e:
    logger.error(f"Unexpected error with GPU symbolic fusion or CuPy: {e}. GPU benchmark will be skipped.")
    fuse_masks_gpu = None
    build_relation_graph_gpu = None


def benchmark_module(module_name: str, module_funcs: dict, masks_list: list[np.ndarray]) -> dict:
    """
    Benchmarks the mask fusion and graph building operations for a given module.

    Args:
        module_name (str): Name of the module (e.g., "CPU", "GPU").
        module_funcs (dict): Dictionary with "fuse" and "graph" functions.
        masks_list (list[np.ndarray]): List of masks to benchmark against.

    Returns:
        dict: A dictionary containing benchmark results (time, nodes, edges).
    """
    if module_funcs["fuse"] is None or module_funcs["graph"] is None:
        logger.warning(f"Skipping benchmark for {module_name} module: functions not available.")
        return {"time": float('nan'), "nodes": 0, "edges": 0}

    start_time = time.time()
    
    # Perform mask fusion
    try:
        fused_mask = module_funcs["fuse"](masks_list)
        logger.debug(f"{module_name} fused mask sum: {fused_mask.sum()}")
    except Exception as e:
        logger.error(f"Error during {module_name} mask fusion: {e}")
        return {"time": float('nan'), "nodes": 0, "edges": 0}

    # Perform graph building
    try:
        graph = module_funcs["graph"](masks_list)
    except Exception as e:
        logger.error(f"Error during {module_name} graph building: {e}")
        return {"time": float('nan'), "nodes": 0, "edges": 0}
    
    end_time = time.time()
    
    return {
        "time": end_time - start_time,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges()
    }

def load_masks_for_benchmark(masks_dir: Path, max_masks_per_shard: int = 100) -> dict:
    """
    Loads masks from a specified directory, grouped by problem ID, for benchmarking.
    It will load up to `max_masks_per_shard` masks per problem to simulate shard data.

    Args:
        masks_dir (Path): Path to the directory containing mask PNGs (e.g., data/sam_masks/train).
        max_masks_per_shard (int): Maximum number of masks to load per problem/shard.

    Returns:
        dict: A dictionary where keys are problem IDs and values are lists of NumPy masks.
    """
    problem_masks = {}
    if not masks_dir.exists():
        logger.error(f"Masks directory not found: {masks_dir}. Cannot load masks for benchmark.")
        return {}

    # Group masks by their problem/image stem
    # Assuming mask names are like 'problemX_imageY_sam_mask_Z.png' or 'imageY_sam_mask_Z.png'
    # We need to extract a unique identifier for grouping.
    # For simplicity, let's group by the part before '_sam_mask_'
    
    mask_files = list(masks_dir.glob("*_sam_mask_*.png"))
    if not mask_files:
        logger.warning(f"No SAM masks found in {masks_dir}. Ensure masks are generated and named correctly.")
        return {}

    # Group masks by their base image stem (e.g., "Bongard_0001_0_0" from "Bongard_0001_0_0_sam_mask_0.png")
    grouped_masks_paths = {}
    for mask_file in mask_files:
        # Example: Bongard_0001_0_0_sam_mask_0.png -> Bongard_0001_0_0
        stem_parts = mask_file.stem.split('_sam_mask_')
        if len(stem_parts) > 1:
            base_stem = stem_parts[0]
            if base_stem not in grouped_masks_paths:
                grouped_masks_paths[base_stem] = []
            grouped_masks_paths[base_stem].append(mask_file)
    
    for base_stem, paths in grouped_masks_paths.items():
        # Limit the number of masks loaded per problem/image for performance
        selected_paths = random.sample(paths, min(len(paths), max_masks_per_shard))
        masks_for_problem = []
        for p in selected_paths:
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks_for_problem.append(mask > 0) # Convert to boolean mask
        if masks_for_problem:
            problem_masks[base_stem] = masks_for_problem
        else:
            logger.warning(f"No valid masks loaded for problem {base_stem} from {masks_dir}.")

    logger.info(f"Loaded masks for {len(problem_masks)} unique problems/images from {masks_dir}.")
    return problem_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark symbolic fusion performance (CPU vs. GPU).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--masks_dir",
        required=True,
        type=Path,
        help="Directory containing SAM-generated masks (e.g., data/sam_masks/train)."
    )
    parser.add_argument(
        "--output_dir",
        default="benchmark_results",
        type=Path,
        help="Directory to save benchmark results (JSON)."
    )
    parser.add_argument(
        "--max_masks_per_problem",
        type=int,
        default=100,
        help="Maximum number of masks to load per problem for benchmarking."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading masks from: {args.masks_dir}")
    all_problem_masks = load_masks_for_benchmark(args.masks_dir, args.max_masks_per_problem)

    if not all_problem_masks:
        logger.error("No masks loaded for benchmarking. Exiting.")
        exit()

    modules_to_benchmark = {}
    if fuse_masks_cpu and build_relation_graph_cpu:
        modules_to_benchmark["CPU"] = {"fuse": fuse_masks_cpu, "graph": build_relation_graph_cpu}
    if fuse_masks_gpu and build_relation_graph_gpu:
        modules_to_benchmark["GPU"] = {"fuse": fuse_masks_gpu, "graph": build_relation_graph_gpu}

    if not modules_to_benchmark:
        logger.error("No valid benchmarking modules (CPU/GPU) are available. Exiting.")
        exit()

    shard_benchmark_results = []

    # Simulate sharding by iterating through loaded problems
    # Each problem's masks can be considered a "shard" for this benchmark
    for problem_stem, masks_for_problem in tqdm(all_problem_masks.items(), desc="Benchmarking problems"):
        if not masks_for_problem:
            logger.warning(f"Skipping empty mask list for problem: {problem_stem}")
            continue

        problem_results = {"problem_stem": problem_stem, "num_masks": len(masks_for_problem)}
        
        for module_name, module_funcs in modules_to_benchmark.items():
            logger.info(f"Benchmarking {module_name} for problem {problem_stem} with {len(masks_for_problem)} masks.")
            results = benchmark_module(module_name, module_funcs, masks_for_problem)
            problem_results[module_name] = results
        
        shard_benchmark_results.append(problem_results)

    # Save overall benchmark results
    output_json_path = args.output_dir / "symbolic_fusion_benchmark_results.json"
    with open(output_json_path, "w") as f:
        json.dump(shard_benchmark_results, f, indent=4)
    logger.info(f"Benchmark results saved to: {output_json_path}")

    print("\nBenchmark complete. You can now use visualize_shard_speed.py to plot the results.")
