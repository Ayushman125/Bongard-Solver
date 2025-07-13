# yolofinetuning/data_pipeline.py

import logging
import torch
from torch.utils.data import DataLoader

# ——— NVIDIA DALI support —————————————————————————————————————
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

    HAS_DALI = True
    logging.getLogger("dali").info("DALI enabled")
except ImportError:
    HAS_DALI = False
    logging.getLogger("dali").warning("DALI not found")

# ——— FFCV support ———————————————————————————————————————————
try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields import RGBImageField, IntField

    HAS_FFCV = True
    logging.getLogger("ffcv").info("FFCV enabled")
except ImportError:
    HAS_FFCV = False
    logging.getLogger("ffcv").warning("FFCV not found")

def get_dali_loader(image_dir: str,
                    batch_size: int,
                    num_threads: int,
                    device_id: int,
                    img_size: int):
    assert HAS_DALI, "Install NVIDIA DALI to use this loader"
    pipe = Pipeline(batch_size, num_threads, device_id, seed=42)
    with pipe:
        jpegs, labels = fn.readers.file(
            file_root=image_dir, random_shuffle=True, name="Reader"
        )
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=img_size, resize_y=img_size)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIGenericIterator(
        pipeline=pipe,
        output_map=["images", "labels"],
        last_batch_policy=LastBatchPolicy.PARTIAL,
        reader_name="Reader",
    )

def get_ffcv_loader(beton_path: str,
                    batch_size: int,
                    num_workers: int):
    assert HAS_FFCV, "Install FFCV to use this loader"
    pipelines = {
        "image": RGBImageField(write_mode=RGBImageField.WriteMode.RGB),
        "label": IntField(),
    }
    return Loader(
        beton_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.RANDOM,
        pipelines=pipelines,
    )

def get_pytorch_loader(dataset,
                       batch_size: int,
                       shuffle: bool = True,
                       num_workers: int = 4,
                       pin_memory: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def prefetch_loader(loader):
    """
    For DALI/FFCV this is a no-op (they prefetch internally).
    For PyTorch loader you could wrap a Prefetcher if desired.
    """
    return loader