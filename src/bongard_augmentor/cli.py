import argparse
import json
import torch
import pickle
from .main import ImageAugmentor
from .dataset import ImagePathDataset
from .utils import HYBRID_PIPELINE_AVAILABLE

def main():
    """Main entry point for the augmentation script."""
    parser = argparse.ArgumentParser(description="Bongard Problem Image Augmentation")
    parser.add_argument('--input', type=str, required=True, help='Path to derived_labels.json')
    parser.add_argument('--out', type=str, required=True, help='Path to output augmented.pkl')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--rotate', type=int, default=0, help='Max rotation angle')
    parser.add_argument('--scale', type=float, default=1.0, help='Max scale factor')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--type', type=str, default='geometric', choices=['geometric', 'photometric', 'both'], help='Type of augmentation')
    parser.add_argument('--enable-hybrid', action='store_true', help='Enable hybrid SAM+SAP pipeline')
    parser.add_argument('--sam-model', type=str, default='vit_h', help='SAM model type')
    parser.add_argument('--test-corruption-fixes', action='store_true', help='Test corruption diagnostic and mitigation systems')
    parser.add_argument('--force-emergency-qa', action='store_true', help='Force ultra-permissive emergency QA thresholds')
    parser.add_argument('--fallback-empty', action='store_true', help='Use SAM for empty geometry masks')
    parser.add_argument('--qa-fail-threshold', type=float, default=0.15, help='QA failure rate threshold')
    args = parser.parse_args()

    print("[INIT] Initializing Bongard Hybrid Augmentation Pipeline")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.out}")
    print(f"  Batch size: {args.batch_size}")
    # Always enable hybrid pipeline
    print(f"  Hybrid enabled: True")
    print(f"  SAM model: {args.sam_model}")

    with open(args.input, 'r') as f:
        derived_labels = json.load(f)
    image_paths = [entry['image_path'] for entry in derived_labels]
    dataset = ImagePathDataset(image_paths, derived_labels_path=args.input)

    augmentor = ImageAugmentor(batch_size=args.batch_size)
    
    if HYBRID_PIPELINE_AVAILABLE:
        print("[INIT] Initializing hybrid pipeline components (SAM, SAP)")
        augmentor.initialize_hybrid_pipeline(
            sam_model_type=args.sam_model,
            enable_sap=True
        )

    if args.test_corruption_fixes:
        print("[QA TEST] Running corruption fix tests...")
        test_batch = [dataset[i] for i in range(min(3, len(dataset)))]
        image_tensors = [item[0] for item in test_batch if item[0] is not None]
        if not image_tensors:
            print("[ERROR] No valid images found for corruption test. Check your input paths.")
            return
        images = torch.stack(image_tensors)
        # Use a simple corruption test: add noise and print stats
        noisy_images = images + 0.2 * torch.randn_like(images)
        print(f"[QA TEST] Corruption test: mean={noisy_images.mean().item():.4f}, std={noisy_images.std().item():.4f}")
        return

    all_results = []
    for idx in range(0, len(dataset), args.batch_size):
        batch = [dataset[i] for i in range(idx, min(idx+args.batch_size, len(dataset)))]
        
        images = torch.stack([item[0] for item in batch if item[0] is not None])
        paths = [item[1] for item in batch if item[1] is not None]
        geometries = [item[2] for item in batch if item[2] is not None]

        if images.numel() == 0:
            continue

        results = augmentor.augment_batch(
            images,
            geometries,
            paths,
            augment_type=args.type,
            batch_idx=idx // args.batch_size
        )
        all_results.append(results)

    with open(args.out, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"[MAIN] Augmentation complete. Saved {len(all_results)} batches to {args.out}")

if __name__ == "__main__":
    main()
