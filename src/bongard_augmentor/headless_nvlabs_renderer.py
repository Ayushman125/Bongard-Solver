"""
Headless NVLabs-compatible renderer that matches the official coordinate system without requiring turtle graphics.
"""
import numpy as np
from PIL import Image, ImageDraw
import cv2
import logging
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

class HeadlessNVLabsRenderer:
    """
    Headless renderer that exactly matches NVLabs coordinate system and image generation.
    Uses matplotlib with the same coordinate system as turtle graphics.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.target_image_size = image_size
        self.canvas_size = 800  # Official NVLabs canvas size (800x800)
        self.coord_range = (-360, 360)  # Official NVLabs coordinate range
        
    def render_vertices_to_image(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Render vertices using matplotlib to match NVLabs turtle graphics coordinate system.
        
        Args:
            vertices: List of (x, y) coordinates in NVLabs coordinate system (-360 to 360)
            
        Returns:
            Binary mask image as numpy array (512x512)
        """
        if len(vertices) < 2:
            return np.zeros(self.target_image_size, dtype=np.uint8)
        
        try:
            # Create matplotlib figure with same setup as turtle graphics
            fig, ax = plt.subplots(figsize=(8, 8))  # 8x8 inch figure
            ax.set_xlim(-360, 360)  # Same as turtle coordinate range
            ax.set_ylim(-360, 360)  # Same as turtle coordinate range
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Set white background (like real Bongard-LOGO images)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Draw the path
            if len(vertices) >= 2:
                # Extract x and y coordinates
                x_coords = [v[0] for v in vertices]
                y_coords = [v[1] for v in vertices]
                
                # Draw lines connecting vertices (matching turtle behavior)
                ax.plot(x_coords, y_coords, 'k-', linewidth=8, solid_capstyle='round', solid_joinstyle='round')
                
                # For closed shapes, ensure proper closure
                if len(vertices) >= 3:
                    distance_to_start = np.sqrt((vertices[-1][0] - vertices[0][0])**2 + 
                                              (vertices[-1][1] - vertices[0][1])**2)
                    if distance_to_start < 30:  # Threshold for closed shape
                        # Close the shape explicitly
                        ax.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 
                               'k-', linewidth=8, solid_capstyle='round')
            
            # Remove all margins and padding
            plt.tight_layout(pad=0)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # Render to canvas
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            
            # Get image data (handle different matplotlib versions)
            try:
                buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            except AttributeError:
                # For newer matplotlib versions
                buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
                buf = buf[:, :, :3]  # Remove alpha channel
            else:
                buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            
            # Convert to grayscale
            gray_image = cv2.cvtColor(buf, cv2.COLOR_RGB2GRAY)
            
            # Resize to target size (512x512) like official NVLabs
            resized_image = cv2.resize(gray_image, self.target_image_size, interpolation=cv2.INTER_AREA)
            
            # Convert to binary mask (white background -> 0, black lines -> 255)
            _, binary_mask = cv2.threshold(resized_image, 240, 255, cv2.THRESH_BINARY_INV)
            
            plt.close(fig)
            
            coverage = np.mean(binary_mask > 0) * 100
            logging.debug(f"Matplotlib NVLabs render: {binary_mask.shape}, coverage={coverage:.1f}%")
            
            return binary_mask
            
        except Exception as e:
            logging.error(f"Error in matplotlib NVLabs rendering: {e}")
            plt.close('all')  # Clean up any remaining figures
            return self._fallback_pil_render(vertices)
    
    def _fallback_pil_render(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Fallback PIL rendering that mimics the NVLabs coordinate system.
        """
        # Create high-res image for better quality
        high_res_size = (self.target_image_size[0] * 4, self.target_image_size[1] * 4)
        
        # Create PIL image with white background
        pil_image = Image.new('L', (high_res_size[1], high_res_size[0]), color=255)
        draw = ImageDraw.Draw(pil_image)
        
        if len(vertices) < 2:
            # Resize to target and convert to binary
            resized = pil_image.resize(self.target_image_size, Image.LANCZOS)
            result = np.array(resized)
            _, binary_mask = cv2.threshold(result, 240, 255, cv2.THRESH_BINARY_INV)
            return binary_mask
        
        # Convert NVLabs coordinates to PIL pixel coordinates
        pil_vertices = []
        for x, y in vertices:
            # Map from (-360, 360) to high-res pixel coordinates
            pixel_x = int((x + 360) / 720 * high_res_size[1])
            pixel_y = int((360 - y) / 720 * high_res_size[0])  # Flip Y axis for PIL
            
            # Clamp to valid range
            pixel_x = max(0, min(high_res_size[1] - 1, pixel_x))
            pixel_y = max(0, min(high_res_size[0] - 1, pixel_y))
            
            pil_vertices.append((pixel_x, pixel_y))
        
        # Draw thick lines (scale line width for high-res)
        line_width = max(32, int(high_res_size[0] * 0.01))  # Thick lines for high-res
        
        if len(pil_vertices) >= 2:
            # Draw lines between consecutive vertices
            for i in range(len(pil_vertices) - 1):
                draw.line([pil_vertices[i], pil_vertices[i + 1]], fill=0, width=line_width)
            
            # For closed shapes, connect last to first
            if len(pil_vertices) >= 3:
                start_point = np.array(pil_vertices[0])
                end_point = np.array(pil_vertices[-1])
                distance = np.linalg.norm(end_point - start_point)
                if distance < line_width * 2:  # Close the shape
                    draw.line([pil_vertices[-1], pil_vertices[0]], fill=0, width=line_width)
        
        # Resize to target size
        resized = pil_image.resize(self.target_image_size, Image.LANCZOS)
        result = np.array(resized)
        
        # Convert to binary mask
        _, binary_mask = cv2.threshold(result, 240, 255, cv2.THRESH_BINARY_INV)
        
        coverage = np.mean(binary_mask > 0) * 100
        logging.debug(f"PIL fallback render: {binary_mask.shape}, coverage={coverage:.1f}%")
        
        return binary_mask


class HeadlessActionMaskGenerator:
    """
    Headless action mask generator using NVLabs-compatible coordinate system.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
        self.renderer = HeadlessNVLabsRenderer(image_size)
        
    def generate_mask_from_vertices(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Generate binary mask from vertices using headless NVLabs-compatible rendering.
        
        Args:
            vertices: List of (x, y) coordinates in NVLabs coordinate system (-360 to 360)
            
        Returns:
            Binary mask as numpy array
        """
        try:
            mask = self.renderer.render_vertices_to_image(vertices)
            
            coverage = np.mean(mask > 0) * 100
            logging.info(f"Generated headless NVLabs mask: {mask.shape}, coverage={coverage:.2f}%")
            
            return mask
            
        except Exception as e:
            logging.error(f"Error generating headless NVLabs mask: {e}")
            return np.zeros(self.image_size, dtype=np.uint8)
