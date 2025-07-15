import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def

@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def dali_augment_pipeline():
    imgs, bboxes, labels = fn.readers.file(file_root="raw_images",
                                           random_shuffle=True,
                                           name="Reader")
    images = fn.decoders.image(imgs, device="mixed")
    images = fn.rotate(images, angle=fn.random.uniform(range=(0.0, 360.0)))
    images = fn.crop_mirror_normalize(images,
                                      crop=(320, 320),
                                      mean=[0.485*255, 0.456*255, 0.406*255],
                                      std=[0.229*255, 0.224*255, 0.225*255])
    # MixUp & CutMix using GpuCoCoMixin or custom ops if available
    return images, bboxes.gpu(), labels.gpu()

def dali_augment(image_paths, labels):
    pipe = dali_augment_pipeline()
    pipe.build()
    for _ in range(len(image_paths) // pipe.batch_size):
        imgs, bbs, lbls = pipe.run()
        # Convert to CPU numpy and adjust YOLO format...
        yield imgs.as_cpu().as_array(), bbs.as_cpu().as_array(), lbls.as_cpu().as_array()
