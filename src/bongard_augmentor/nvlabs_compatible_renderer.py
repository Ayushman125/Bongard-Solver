"""
NVLabs-compatible image renderer that matches the official Bongard-LOGO coordinate system and image generation.
This renderer uses the same turtle graphics approach as the official NVLabs implementation.
"""
import os
import tempfile
import turtle
import numpy as np
from PIL import Image
import logging
from typing import List, Tuple, Optional
import cv2

class NVLabsCompatibleRenderer:
    """
    Renderer that exactly matches the official NVLabs Bongard-LOGO coordinate system and image generation process.
    Uses turtle graphics with 800x800 canvas and (-360, 360) coordinate range, then resizes to 512x512.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.target_image_size = image_size
        self.canvas_size = (800, 800)  # Official NVLabs canvas size
        self.coord_range = (-360, 360)  # Official NVLabs coordinate range
        self.screen = None
        self.turtle = None
        
    def setup_turtle_environment(self):
        """Set up turtle graphics environment exactly like official NVLabs."""
        try:
            # Create screen with official NVLabs settings
            self.screen = turtle.Screen()
            width, height = self.canvas_size
            self.screen.setup(width=width, height=height)
            self.screen.screensize(width, height)
            self.screen.bgcolor("white")  # White background like real images
            
            # Create turtle with official settings
            self.turtle = turtle.Turtle()
            self.turtle.pen(fillcolor="white", pencolor="black", pendown=False, pensize=8, speed=0)
            self.turtle.hideturtle()
            
            # Disable animation for faster rendering
            self.screen.tracer(0, 0)
            
            logging.info(f"Turtle environment setup: canvas={self.canvas_size}, range={self.coord_range}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup turtle environment: {e}")
            return False
    
    def cleanup_turtle_environment(self):
        """Clean up turtle graphics environment."""
        try:
            if self.screen:
                self.screen.bye()
            self.screen = None
            self.turtle = None
        except Exception as e:
            logging.warning(f"Error during turtle cleanup: {e}")
    
    def render_vertices_to_image(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Render vertices to image using official NVLabs coordinate system and approach.
        
        Args:
            vertices: List of (x, y) coordinates in NVLabs coordinate system (-360 to 360)
            
        Returns:
            Binary mask image as numpy array (512x512)
        """
        if len(vertices) < 2:
            return np.zeros(self.target_image_size, dtype=np.uint8)
        
        try:
            # Setup turtle environment
            if not self.setup_turtle_environment():
                return self._fallback_opencv_render(vertices)
            
            # Clear canvas
            self.turtle.clear()
            
            # Convert vertices to turtle commands
            self._draw_vertices_with_turtle(vertices)
            
            # Update screen to ensure all drawing is complete
            self.screen.update()
            
            # Save to temporary file and load as image
            image_array = self._capture_turtle_canvas()
            
            # Cleanup
            self.cleanup_turtle_environment()
            
            return image_array
            
        except Exception as e:
            logging.error(f"Error in NVLabs-compatible rendering: {e}")
            self.cleanup_turtle_environment()
            return self._fallback_opencv_render(vertices)
    
    def _draw_vertices_with_turtle(self, vertices: List[Tuple[float, float]]):
        """Draw vertices using turtle graphics, matching official NVLabs approach."""
        if len(vertices) < 2:
            return
        
        # Start from first vertex
        self.turtle.penup()
        start_x, start_y = vertices[0]
        self.turtle.goto(start_x, start_y)
        self.turtle.pendown()
        
        # Draw lines to all other vertices
        for i in range(1, len(vertices)):
            x, y = vertices[i]
            self.turtle.goto(x, y)
        
        # For closed shapes, return to start
        if len(vertices) >= 3:
            distance_to_start = np.sqrt((vertices[-1][0] - vertices[0][0])**2 + 
                                      (vertices[-1][1] - vertices[0][1])**2)
            if distance_to_start < 30:  # Threshold for closed shape
                self.turtle.goto(vertices[0][0], vertices[0][1])
        
        logging.debug(f"Drew {len(vertices)} vertices with turtle graphics")
    
    def _capture_turtle_canvas(self) -> np.ndarray:
        """Capture turtle canvas and convert to numpy array, matching official NVLabs process."""
        try:
            # Create temporary file for PostScript output
            with tempfile.NamedTemporaryFile(suffix='.ps', delete=False) as tmp_ps:
                ps_filepath = tmp_ps.name
            
            # Save canvas as PostScript (like official NVLabs)
            self.screen.getcanvas().postscript(file=ps_filepath)
            
            # Load PostScript and resize to 512x512 (exactly like official NVLabs)
            pil_image = Image.open(ps_filepath)
            pil_image = pil_image.resize(self.target_image_size)
            
            # Convert to grayscale numpy array
            image_array = np.array(pil_image.convert('L'))
            
            # Convert to binary mask (white background -> 0, black drawings -> 255)
            _, binary_mask = cv2.threshold(image_array, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Clean up temporary file
            try:
                os.unlink(ps_filepath)
            except:
                pass
            
            logging.debug(f"Captured turtle canvas: {binary_mask.shape}, coverage={np.mean(binary_mask > 0)*100:.1f}%")
            return binary_mask
            
        except Exception as e:
            logging.error(f"Failed to capture turtle canvas: {e}")
            return np.zeros(self.target_image_size, dtype=np.uint8)
    
    def _fallback_opencv_render(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Fallback OpenCV rendering that mimics the NVLabs coordinate system.
        """
        mask = np.zeros(self.target_image_size, dtype=np.uint8)
        
        if len(vertices) < 2:
            return mask
        
        # Convert NVLabs coordinates (-360, 360) to pixel coordinates (0, 512)
        pixel_vertices = []
        for x, y in vertices:
            # Map from (-360, 360) to (0, 512)
            pixel_x = int((x + 360) / 720 * self.target_image_size[1])
            pixel_y = int((360 - y) / 720 * self.target_image_size[0])  # Flip Y axis
            
            # Clamp to valid range
            pixel_x = max(0, min(self.target_image_size[1] - 1, pixel_x))
            pixel_y = max(0, min(self.target_image_size[0] - 1, pixel_y))
            
            pixel_vertices.append([pixel_x, pixel_y])
        
        if len(pixel_vertices) >= 2:
            points_array = np.array(pixel_vertices, dtype=np.int32)
            
            # Use thick lines to match official appearance
            thickness = max(8, int(self.target_image_size[0] * 0.015))  # ~1.5% of image height
            cv2.polylines(mask, [points_array], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
            
            # Add end caps
            cap_radius = thickness // 2
            for point in pixel_vertices:
                cv2.circle(mask, tuple(point), cap_radius, 255, -1)
        
        logging.debug(f"Fallback OpenCV render: {len(vertices)} vertices, coverage={np.mean(mask > 0)*100:.1f}%")
        return mask


class ActionMaskGeneratorNVLabsCompatible:
    """
    NVLabs-compatible action mask generator that uses the official coordinate system and rendering approach.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
        self.renderer = NVLabsCompatibleRenderer(image_size)
        
    def generate_mask_from_vertices(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Generate binary mask from vertices using NVLabs-compatible rendering.
        
        Args:
            vertices: List of (x, y) coordinates in NVLabs coordinate system (-360 to 360)
            
        Returns:
            Binary mask as numpy array
        """
        try:
            mask = self.renderer.render_vertices_to_image(vertices)
            
            coverage = np.mean(mask > 0) * 100
            logging.info(f"Generated NVLabs-compatible mask: {mask.shape}, coverage={coverage:.2f}%")
            
            return mask
            
        except Exception as e:
            logging.error(f"Error generating NVLabs-compatible mask: {e}")
            return np.zeros(self.image_size, dtype=np.uint8)
    
    def generate_mask_from_action_commands(self, action_commands: List[str]) -> np.ndarray:
        """
        Generate mask from action commands by parsing them and extracting vertices.
        This method needs to be integrated with the action parser.
        """
        # This would integrate with the existing parser to get vertices from commands
        # For now, return empty mask
        logging.warning("generate_mask_from_action_commands not yet implemented for NVLabs renderer")
        return np.zeros(self.image_size, dtype=np.uint8)
