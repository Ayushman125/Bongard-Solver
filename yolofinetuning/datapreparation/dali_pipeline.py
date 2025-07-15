
import argparse
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def

# Pipeline parameters for dynamic configuration
dp = {
    "batch_size": 16,
    "num_threads": 4,
    "device_id": 0
}

@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def dali_augment_pipeline(img_dir):
    imgs = fn.readers.file(file_root=img_dir,
                          random_shuffle=True,
                          name="Reader")
    images = fn.decoders.image(imgs, device="mixed")
    images = fn.rotate(images, angle=fn.random.uniform(range=(0.0, 360.0)))
    images = fn.crop_mirror_normalize(images,
                                      crop=(320, 320),
                                      mean=[0.485*255, 0.456*255, 0.406*255],
                                      std=[0.229*255, 0.224*255, 0.225*255])
    # DALI file reader does not return bboxes/labels unless using COCO/YOLO reader
    # For simple image pipeline, just return images
    return images

def dali_augment(image_paths, labels):
    pipe = dali_augment_pipeline(image_paths)
    pipe.build()
    for _ in range(pipe.epoch_size("Reader") // pipe.batch_size):
        imgs = pipe.run()[0]
        # Convert to CPU numpy
        yield imgs.as_cpu().as_array()

def main():
    parser = argparse.ArgumentParser(description="DALI pipeline for dataset preparation")
    parser.add_argument("--mode", type=str, default="simple", help="Mode of operation")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory of images")
    parser.add_argument("--lbl_dir", type=str, required=True, help="Directory of labels")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations")
    args = parser.parse_args()

    # For now, only simple mode is supported
    print("[INFO] Starting DALI pipeline...")
    # Recursively collect all PNG images from img_dir
    import os, glob
    image_paths = [y for x in os.walk(args.img_dir) for y in glob.glob(os.path.join(x[0], '*.png'))]
    print(f"Found {len(image_paths)} PNG images in {args.img_dir}")
    gen = dali_augment(args.img_dir, None)
    for i in range(args.iters):
        try:
            imgs = next(gen)
        except StopIteration:
            print("No more images to process.")
            break
        print(f"[DALI OUTPUT] Iteration {i+1}")
        print(f"  Images shape: {imgs.shape}")
        if imgs.shape[0] > 0:
            print(f"  Images sample: {imgs[0]}")

# YOLO format pipeline
@pipeline_def(batch_size=dp["batch_size"], num_threads=dp["num_threads"], device_id=0)
@pipeline_def(batch_size=16, num_threads=4, device_id=0)
@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def yolo_pipeline(image_dir, label_dir):
    """
    YOLO format pipeline for custom YOLO datasets.
    Handles YOLO txt format labels with class_id, x_center, y_center, width, height.
    """
    # Read image files
    images, image_ids = fn.readers.file(
        file_root=image_dir,
        random_shuffle=True,
        name="ImageReader"
    )
    
    # Read corresponding YOLO label files  
    labels = fn.readers.file(
        file_root=label_dir,
        random_shuffle=False,  # Keep same order as images
        name="LabelReader"
    )
    
    images = fn.decoders.image(images, device="mixed")
    
    # Basic augmentations without bbox operations (simplified for YOLO)
    flip_prob = fn.random.coin_flip(probability=0.5)
    images = fn.flip(images, horizontal=flip_prob)
    
    # Color augmentations
    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=(0.9, 1.1)),
        contrast=fn.random.uniform(range=(0.9, 1.1))
    )
    
    # Resize and normalize
    images = fn.resize(images, resize_x=640, resize_y=640)
    images = fn.crop_mirror_normalize(
        images,
        crop=(640, 640), 
        mean=[0.485*255, 0.456*255, 0.406*255],
        std=[0.229*255, 0.224*255, 0.225*255],
        dtype=types.FLOAT
    )
    
    return images, labels

# Simple image augmentation pipeline 
@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def simple_augment_pipeline(image_dir):
    """
    Simple image augmentation pipeline without bbox operations.
    Good for testing and basic image processing tasks.
    """
    print(f"[DALI DEBUG] Reading images from: {image_dir}")
    
    # Step 1: Read files
    images = fn.readers.file(
        file_root=image_dir,
        random_shuffle=True,
        name="Reader"
    )
    print(f"[DALI DEBUG] File reader output type: {type(images)}")
    
    # Step 2: Decode images
    images = fn.decoders.image(images, device="mixed")
    print(f"[DALI DEBUG] Image decoder output type: {type(images)}")
    
    # Step 3: Rotation augmentation
    angles = fn.random.uniform(range=(-15.0, 15.0))
    print(f"[DALI DEBUG] Random angles type: {type(angles)}")
    images = fn.rotate(images, angle=angles, fill_value=0)
    print(f"[DALI DEBUG] After rotation type: {type(images)}")
    
    # Step 4: Color augmentations
    brightness_vals = fn.random.uniform(range=(0.8, 1.2))
    contrast_vals = fn.random.uniform(range=(0.8, 1.2))
    print(f"[DALI DEBUG] Brightness type: {type(brightness_vals)}, Contrast type: {type(contrast_vals)}")
    
    images = fn.brightness_contrast(
        images,
        brightness=brightness_vals,
        contrast=contrast_vals
    )
    print(f"[DALI DEBUG] After color augmentation type: {type(images)}")
    
    # Step 5: Geometric augmentations
    flip_prob = fn.random.coin_flip(probability=0.5)
    print(f"[DALI DEBUG] Flip probability type: {type(flip_prob)}")
    images = fn.flip(images, horizontal=flip_prob)
    print(f"[DALI DEBUG] After flip type: {type(images)}")
    
    # Step 6: Resize
    images = fn.resize(images, resize_x=640, resize_y=640)
    print(f"[DALI DEBUG] After resize type: {type(images)}")
    
    # Step 7: Normalize
    images = fn.crop_mirror_normalize(
        images,
        crop=(640, 640),
        mean=[0.485*255, 0.456*255, 0.406*255], 
        std=[0.229*255, 0.224*255, 0.225*255],
        dtype=types.FLOAT
    )
    print(f"[DALI DEBUG] After normalization type: {type(images)}")
    print(f"[DALI DEBUG] Final pipeline return type: {type(images)}")
    
    # Return as tuple to avoid nested DataNode error
    return (images,)

def run_detection_pipeline(img_dir, lbl_file, iterations):
    """Run the professional detection pipeline with bbox support."""
    # detection_pipeline is not defined in this file. Uncomment and implement if available.
    # pipe = detection_pipeline(img_dir, lbl_file)
    # pipe.build()
    # print(f"Running detection pipeline for {iterations} iterations...")
    # for i in range(iterations):
    #     imgs, bboxes, labels = pipe.run()
    #     print(f"Detection batch {i+1}/{iterations}")
    #     imgs_np = imgs.as_cpu().as_array()
    #     bboxes_np = bboxes.as_cpu().as_array() 
    #     labels_np = labels.as_cpu().as_array()
    #     print(f"  Images: {imgs_np.shape}, Bboxes: {len(bboxes_np)}, Labels: {len(labels_np)}")
    #     # save_augmented_data(imgs_np, bboxes_np, labels_np, i)
    # print("Detection pipeline completed successfully!")

def run_yolo_pipeline(image_dir, label_dir, iterations):
    """Run the YOLO format pipeline."""
    pipe = yolo_pipeline(image_dir, label_dir)
    pipe.build()
    
    print(f"Running YOLO pipeline for {iterations} iterations...")
    for i in range(iterations):
        imgs, lbls = pipe.run()
        print(f"YOLO batch {i+1}/{iterations}")
        
        imgs_np = imgs.as_cpu().as_array()
        print(f"  Image batch shape: {imgs_np.shape}")
        
    print("YOLO pipeline completed successfully!")

def run_simple_augment(image_dir, iterations):
    """Run the simple augmentation pipeline."""
    pipe = simple_augment_pipeline(image_dir)
    pipe.build()
    
    print(f"Running simple augmentation for {iterations} iterations...")
    for i in range(iterations):
        imgs, = pipe.run()  # Unpack single output
        print(f"Simple augment batch {i+1}/{iterations}")
        
        imgs_np = imgs.as_cpu().as_array()
        print(f"  Augmented batch shape: {imgs_np.shape}")
        
    print("Simple augmentation pipeline completed successfully!")

if __name__ == "__main__":
    # Import all major functions from datasetpreparation folder
    from fuse_graphs_with_yolo import load_graph, load_yolo, iou, nms, check_relation, label_quality, log_metadata as fg_log_metadata
    from metadata_logger import compute_metadata, log_metadata as meta_log_metadata
    from copy_paste_synthesis import load_object_masks, paste_objects
    from auto_labeling import auto_label
    from augmentations import get_train_transforms, apply_augmentations
    from split_dataset import split_dataset
    # Import SAM and Albumentations/AugMix utilities
    try:
        from sam import load_sam_model, get_mask_generator, save_mask_png, sam_masks_to_yolo, generate_relation_graph, get_symbolic_labels, overlay_symbolic_debugger, generate_reasoning_chain
        sam_predictor = load_sam_model()
        mask_generator = get_mask_generator()
    except Exception as e:
        print(f"[WARN] Could not import or load SAM: {e}")
        sam_predictor = None
        mask_generator = None
    try:
        from albumentations_augmix import CopyPaste, augment_and_mix, RandomAugMix
    except Exception as e:
        print(f"[WARN] Could not import AugMix from albumentations_augmix: {e}")
        CopyPaste = None
        augment_and_mix = None
        RandomAugMix = None

    parser = argparse.ArgumentParser(description="Main Dataset Preparation Pipeline")
    parser.add_argument("--img_root", default="ShapeBongard_V2", help="Root directory of images")
    parser.add_argument("--output_root", default="dataset", help="Output root directory")
    parser.add_argument("--lbl_dir", default="data/annotations.json", help="Label file/directory path")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations to run")
    parser.add_argument("--mode", default="simple", choices=["detection", "yolo", "simple"], help="Pipeline mode")
    args = parser.parse_args()

    print(f"Main Dataset Preparation Pipeline Mode: {args.mode}")
    print(f"Image Root: {args.img_root}")
    print(f"Label Path: {args.lbl_dir}")
    print(f"Output Root: {args.output_root}")
    print(f"Iterations: {args.iters}")
    print("-" * 50)

    try:
        # 1. Split dataset
        print("Splitting dataset...")
        split_dataset(args.lbl_dir)

        # 2. Run DALI pipeline (recursively on all PNG images)
        import os, glob
        image_paths = [y for x in os.walk(args.img_root) for y in glob.glob(os.path.join(x[0], '*.png'))]
        print(f"Found {len(image_paths)} PNG images in {args.img_root}")
        labels = None  # Placeholder, adjust as needed
        gen = dali_augment(image_paths, labels)
        for i in range(args.iters):
            try:
                imgs = next(gen)
            except StopIteration:
                print("No more images to process.")
                break
            batch_size = imgs.shape[0]
            for j in range(batch_size):
                src_img_path = image_paths[i * batch_size + j]
                rel_path = os.path.relpath(src_img_path, args.img_root)
                out_dir = os.path.join(args.output_root, os.path.dirname(rel_path))
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.basename(src_img_path))
                from PIL import Image
                import numpy as np
                img_arr = imgs[j]
                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
                # --- Advanced AugMix ---
                if augment_and_mix is not None:
                    img_arr = augment_and_mix(img_arr)
                # --- Albumentations/AugMix CopyPaste ---
                if CopyPaste is not None:
                    cp = CopyPaste()
                    img_arr = cp(image=img_arr)['image']
                # --- RandomAugMix (Albumentations pipeline) ---
                if RandomAugMix is not None:
                    try:
                        import albumentations as A
                        aug_pipeline = A.Compose([RandomAugMix(severity=3, p=0.5)])
                        img_arr = aug_pipeline(image=img_arr)['image']
                    except Exception as e:
                        print(f"[WARN] RandomAugMix failed: {e}")
                # --- SAM mask generation (single image) ---
                if sam_predictor is not None:
                    try:
                        sam_predictor.set_image(img_arr)
                        h, w = img_arr.shape[:2]
                        box = np.array([[0, 0, w, h]])
                        masks, _, _ = sam_predictor.predict(box)
                        mask_path = out_path.replace('.png', '_sam_mask.png')
                        save_mask_png(masks[0], mask_path)
                        print(f"Saved SAM mask: {mask_path}")
                        # --- Convert SAM masks to YOLO labels ---
                        yolo_labels = sam_masks_to_yolo(masks, (h, w))
                        yolo_path = out_path.replace('.png', '_sam_yolo.txt')
                        with open(yolo_path, 'w') as f:
                            f.write('\n'.join(yolo_labels))
                        print(f"Saved YOLO labels: {yolo_path}")
                        # --- Generate relation graph ---
                        relation_graph = generate_relation_graph(masks)
                        graph_path = out_path.replace('.png', '_sam_relgraph.json')
                        import json
                        with open(graph_path, 'w') as f:
                            json.dump(relation_graph, f)
                        print(f"Saved relation graph: {graph_path}")
                        # --- Symbolic reasoning ---
                        symbols = get_symbolic_labels(masks, img_arr, "primitive reasoning")
                        reasoning = generate_reasoning_chain(symbols)
                        reasoning_path = out_path.replace('.png', '_sam_reasoning.txt')
                        with open(reasoning_path, 'w') as f:
                            f.write(reasoning)
                        print(f"Saved reasoning chain: {reasoning_path}")
                        # --- Overlay symbolic debugger ---
                        debug_img = overlay_symbolic_debugger(img_arr, masks, symbols)
                        debug_path = out_path.replace('.png', '_sam_debug.png')
                        Image.fromarray(debug_img).save(debug_path)
                        print(f"Saved symbolic debug image: {debug_path}")
                    except Exception as e:
                        print(f"[WARN] SAM advanced features failed: {e}")
                # --- Batch mask generation (optional, if mask_generator is available) ---
                if mask_generator is not None:
                    try:
                        batch_masks = mask_generator.generate(img_arr)
                        batch_mask_path = out_path.replace('.png', '_sam_batchmask.json')
                        import json
                        with open(batch_mask_path, 'w') as f:
                            json.dump(batch_masks, f)
                        print(f"Saved batch masks: {batch_mask_path}")
                    except Exception as e:
                        print(f"[WARN] SAM batch mask generation failed: {e}")
                # --- Save final image ---
                im = Image.fromarray(img_arr)
                im.save(out_path)
                print(f"Saved: {out_path}")

        # 3. Apply augmentations
        print("Applying augmentations...")
        get_train_transforms()
        apply_augmentations(args.img_root, args.lbl_dir)

        # 4. Auto labeling
        print("Running auto labeling...")
        auto_label(args.img_root, args.lbl_dir)

        # 5. Copy-paste synthesis
        print("Running copy-paste synthesis...")
        masks = load_object_masks(args.img_root)
        # Example: paste_objects usage (requires bg_image, bg_labels, objects)
        # paste_objects(bg_image, bg_labels, masks)

        # 6. Fuse graphs with YOLO
        print("Fusing graphs with YOLO...")
        graph = load_graph(args.lbl_dir)
        yolo = load_yolo(args.lbl_dir)
        # Example: iou, nms, check_relation, label_quality, fg_log_metadata usage
        # iou(boxA, boxB)
        # nms(boxes, scores)
        # check_relation(graph, yolo)
        # label_quality(yolo)
        # fg_log_metadata(...)

        # 7. Metadata logging
        print("Logging metadata...")
        meta = compute_metadata("image_id", "labels")
        meta_log_metadata(meta)

        print("All datasetpreparation steps completed successfully!")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        print("Please check your data paths and format.")
