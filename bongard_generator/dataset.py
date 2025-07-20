from PIL import Image, ImageDraw, ImageFilter
from .shape_renderer import draw_shape
# from .style import apply_noise, apply_checker # Assuming you have this file

# Mock style functions if they don't exist
def apply_noise(img, cfg):
    # Replace with your actual noise implementation
    return img

def apply_checker(img, cfg):
    # Replace with your actual checkerboard implementation
    return img

def create_composite_scene(objects, cfg):
    img = Image.new("RGB",(cfg.canvas_size,cfg.canvas_size),"white")
    draw = ImageDraw.Draw(img)

    for obj in objects:
        if obj.get('prototype') and 'prototype_action' in obj:
            obj['prototype_action'].draw(img, obj['center'], obj['size'], cfg)
        else:
            draw_shape(draw, obj, cfg)

    # background texture
    if cfg.bg_texture=='noise':
        img = apply_noise(img, cfg)
    elif cfg.bg_g_texture=='checker':
        img = apply_checker(img, cfg)

    # GAN stylization
    if hasattr(cfg, 'styler') and cfg.generator.use_gan and cfg.styler:
        img = cfg.styler.stylize(img)

    # final binarize
    img = img.convert("L").filter(ImageFilter.GaussianBlur(0.5))
    return img.point(lambda p:255 if p>128 else 0,'1')
