import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
from PIL import Image, ImageTk
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable


@dataclass
class Filter:
    """Represents a filter applied to an image"""
    id: str  # Unique identifier
    name: str  # Display name
    type: str  # Filter type
    preview: bool = False  # Flag for preview mode
    params: Dict[str, Any] = field(default_factory=dict)  # Parameters for the filter
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class FilterStack:
    """Manages a stack of filters applied to an image"""
    
    def __init__(self):
        self.filters: List[Filter] = []
    
    def add_filter(self, name: str, filter_type: str, params: Dict[str, Any] = None,preview=False) -> Filter:
        """Add a filter to the stack and return it"""
        if params is None:
            params = {}
        
        filter_id = str(uuid.uuid4())
        new_filter = Filter(id=filter_id, name=name, type=filter_type, params=params,preview=preview)
        if len(self.filters) > 0 and self.filters[-1].preview:
            self.filters[-1] = new_filter
        else:
            self.filters.append(new_filter)
        return new_filter
    
    def remove_filter(self, filter_id: str) -> bool:
        """Remove a filter by ID"""
        for i, filter_obj in enumerate(self.filters):
            if filter_obj.id == filter_id:
                self.filters.pop(i)
                return True
        return False
    
    def clear(self) -> None:
        """Clear all filters"""
        self.filters.clear()
    
    def get_filter(self, filter_id: str) -> Optional[Filter]:
        """Get a filter by ID"""
        for filter_obj in self.filters:
            if filter_obj.id == filter_id:
                return filter_obj
        return None
    
    def get_filters(self) -> List[Filter]:
        """Get all filters"""
        return self.filters
    
    def move_filter(self, filter_id: str, new_position: int) -> bool:
        """Move a filter to a new position"""
        if new_position < 0 or new_position >= len(self.filters):
            return False
        
        for i, filter_obj in enumerate(self.filters):
            if filter_obj.id == filter_id:
                filter_obj = self.filters.pop(i)
                self.filters.insert(new_position, filter_obj)
                return True
        
        return False


class ImageProcessor:
    """Class to handle image processing operations"""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.current_filename = None
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Keep track of original dimensions
        self.original_dimensions = (0, 0)
        
        # Filter stack to track applied filters
        self.filter_stack = FilterStack()
        
        # Map of filter types to processing functions
        self.filter_functions = {
            'grayscale': self._apply_grayscale,
            'resize': self._apply_resize,
            'crop': self._apply_crop,
            'flip': self._apply_flip,
            'rotate': self._apply_rotate,
            'brightness_contrast': self._apply_brightness_contrast,
            'color_balance': self._apply_color_balance,
            'hue_saturation': self._apply_hue_saturation,
            'gaussian_blur': self._apply_gaussian_blur,
            'noise_reduction': self._apply_noise_reduction,
            'histogram_equalization': self._apply_histogram_equalization,
            'edge_detection': self._apply_edge_detection,
            'shear': self._apply_shear
        }
    
    def load_image(self, file_path):
        """Load an image from the given file path"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                raise ValueError("Failed to load image")
            
            self.original_dimensions = self.original_image.shape[:2]
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.current_filename = file_path
            return True
        except Exception as e:
            raise e
    
    def save_image(self, file_path):
        """Save the processed image to the given file path"""
        try:
            if self.processed_image is None:
                raise ValueError("No image to save")
            
            save_img = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            result = cv2.imwrite(file_path, save_img)
            
            if not result:
                raise ValueError(f"Failed to save image to {file_path}")
            
            return True
        except Exception as e:
            raise e
    
    def get_display_image(self, width=None, height=None, image=None):
        """Get image formatted for display, optionally resized"""
        if image is None:
            image = self.processed_image
            
        if image is None:
            return None
        
        img = image.copy()
        
        if width is not None and height is not None:
            h, w = img.shape[:2]
            aspect = w / h
            
            if width / height > aspect:
                new_width = int(height * aspect)
                new_height = height
            else:
                new_width = width
                new_height = int(width / aspect)
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return Image.fromarray(img)
    
    def apply_filters(self):
        """Apply all filters in the stack to the original image"""
        if self.original_image is None:
            raise ValueError("No image loaded")
        
        self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        for filter_obj in self.filter_stack.get_filters():
            filter_func = self.filter_functions.get(filter_obj.type)
            if filter_func:
                filter_func(filter_obj.params)
            else:
                print(f"Warning: Unknown filter type '{filter_obj.type}'")
        
        # recalcualte histogram after applying filters
        

        return True
    
    def add_filter(self, name: str, filter_type: str, params: Dict[str, Any] = None,preview=False) -> Filter:
        """Add a filter to the stack and apply it"""
        if params is None:
            params = {}
        new_filter = self.filter_stack.add_filter(name, filter_type, params,preview=preview)
        self.apply_filters()        
        return new_filter
    
    def remove_filter(self, filter_id: str) -> bool:
        """Remove a filter and update the image"""
        result = self.filter_stack.remove_filter(filter_id)
        if result:
            self.apply_filters()
        return result
    
    def clear_filters(self) -> None:
        """Clear all filters and reset to original image"""
        self.filter_stack.clear()
        if self.original_image is not None:
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
    
    def move_filter(self, filter_id: str, new_position: int) -> bool:
        """Move a filter and update the image"""
        result = self.filter_stack.move_filter(filter_id, new_position)
        if result:
            self.apply_filters()
        return result
    
    def _apply_grayscale(self, params: Dict[str, Any]) -> None:
        """Apply grayscale filter"""
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
        self.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    def _apply_resize(self, params: Dict[str, Any]) -> None:
        """Apply resize filter"""
        width_percent = params.get('width_percent', 100)
        height_percent = params.get('height_percent', 100)
        
        height, width = self.processed_image.shape[:2]
        new_width = int(width * width_percent / 100)
        new_height = int(height * height_percent / 100)
        
        self.processed_image = cv2.resize(
            self.processed_image, 
            (new_width, new_height), 
            interpolation=cv2.INTER_AREA
        )
    
    def _apply_crop(self, params: Dict[str, Any]) -> None:
        """Apply crop filter"""
        x = params.get('x', 0)
        y = params.get('y', 0)
        width = params.get('width', 0)
        height = params.get('height', 0)
        
        img_height, img_width = self.processed_image.shape[:2]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return
        if x + width > img_width or y + height > img_height:
            return
        
        self.processed_image = self.processed_image[y:y+height, x:x+width]
    
    def _apply_flip(self, params: Dict[str, Any]) -> None:
        """Apply flip filter"""
        flip_code = params.get('flip_code', 1)
        self.processed_image = cv2.flip(self.processed_image, flip_code)
    
    def _apply_rotate(self, params: Dict[str, Any]) -> None:
        """Apply rotation filter"""
        angle = params.get('angle', 0)
        scale = params.get('scale', 1.0)
        
        height, width = self.processed_image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        self.processed_image = cv2.warpAffine(
            self.processed_image,
            rotation_matrix,
            (width, height)
        )
    
    def _apply_brightness_contrast(self, params: Dict[str, Any]) -> None:
        """Apply brightness and contrast adjustment"""
        brightness = params.get('brightness', 0)
        contrast = params.get('contrast', 1.0)
        
        self.processed_image = cv2.convertScaleAbs(
            self.processed_image, 
            alpha=float(contrast), 
            beta=float(brightness)
        )
    
    def _apply_color_balance(self, params: Dict[str, Any]) -> None:
        """Apply color balance adjustment"""
        r_factor = params.get('r_factor', 1.0)
        g_factor = params.get('g_factor', 1.0)
        b_factor = params.get('b_factor', 1.0)
        
        b, g, r = cv2.split(self.processed_image)
        r = cv2.convertScaleAbs(r, alpha=r_factor)
        g = cv2.convertScaleAbs(g, alpha=g_factor)
        b = cv2.convertScaleAbs(b, alpha=b_factor)
        
        self.processed_image = cv2.merge([b, g, r])
    
    def _apply_hue_saturation(self, params: Dict[str, Any]) -> None:
        """Apply hue and saturation adjustment"""
        hue_shift = params.get('hue_shift', 0)
        saturation_factor = params.get('saturation_factor', 1.0)
        
        hsv_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2HSV)

        # Convert to int16 before shifting to allow negative values
        hue = hsv_image[:, :, 0].astype(np.int16)
        hue = np.mod(hue + hue_shift, 180).astype(np.uint8)
        hsv_image[:, :, 0] = hue

        # Adjust saturation
        hsv_image[:, :, 1] = cv2.convertScaleAbs(hsv_image[:, :, 1], alpha=saturation_factor)

        # Convert back to RGB
        self.processed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    
    def _apply_gaussian_blur(self, params: Dict[str, Any]) -> None:
        """Apply Gaussian blur"""
        kernel_size = params.get('kernel_size', 5)
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.processed_image = cv2.GaussianBlur(
            self.processed_image, 
            (kernel_size, kernel_size), 
            0
        )
    
    def _apply_noise_reduction(self, params: Dict[str, Any]) -> None:
        """Apply noise reduction"""
        strength = params.get('strength', 10)
        
        h = strength
        h_color = strength
        template_window_size = 7
        search_window_size = 21
        
        self.processed_image = cv2.fastNlMeansDenoisingColored(
            self.processed_image,
            None,
            h=h,
            hColor=h_color,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    
    def _apply_histogram_equalization(self, params: Dict[str, Any]) -> None:
        """Apply histogram equalization"""
        if len(self.processed_image.shape) > 2 and self.processed_image.shape[2] == 3:
            yuv_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2YUV)
            yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
            self.processed_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
        else:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            equalized = cv2.equalizeHist(gray)
            self.processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    
    def _apply_edge_detection(self, params: Dict[str, Any]) -> None:
        """Apply edge detection"""
        method = params.get('method', 'sobel')
        threshold1 = params.get('threshold1', 100)
        threshold2 = params.get('threshold2', 200)
        
        if len(self.processed_image.shape) == 3:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.processed_image.copy()
        
        if method == 'sobel':
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
            self.processed_image = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
        elif method == 'laplacian':
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)
            self.processed_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
        elif method == 'canny':
            edges = cv2.Canny(gray, threshold1, threshold2)
            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    def _apply_shear(self, params: Dict[str, Any]) -> None:
        """Apply shear transformation"""
        shear_x = params.get('shear_x', 0.0)
        shear_y = params.get('shear_y', 0.0)
        
        rows, cols = self.processed_image.shape[:2]
        
        M_x = np.float32([
            [1, shear_x, 0],
            [0, 1, 0]
        ])
        
        img_x_sheared = cv2.warpAffine(self.processed_image, M_x, (cols, rows))
        
        M_y = np.float32([
            [1, 0, 0],
            [shear_y, 1, 0]
        ])
        
        self.processed_image = cv2.warpAffine(img_x_sheared, M_y, (cols, rows))
    
    def apply_edge_detection(self, method='sobel', threshold1=100, threshold2=200, preview=False):
        """Apply edge detection and add to filter stack"""
        params = {
            'method': method,
            'threshold1': threshold1,
            'threshold2': threshold2
        }
        
        method_name = {
            'sobel': 'Sobel',
            'laplacian': 'Laplacian',
            'canny': 'Canny'
        }.get(method, 'Edge')
        
        filter_name = f"{method_name} Edge Detection"
        return self.add_filter(filter_name, 'edge_detection', params,preview=preview)
    
    def apply_shear(self, shear_x=0.0, shear_y=0.0):
        """Apply shear transformation and add to filter stack"""
        params = {
            'shear_x': shear_x,
            'shear_y': shear_y
        }
        
        direction = []
        if shear_x != 0:
            direction.append('X')
        if shear_y != 0:
            direction.append('Y')
        
        direction_str = ' & '.join(direction) if direction else 'X/Y'
        filter_name = f"Shear {direction_str}"
        return self.add_filter(filter_name, 'shear', params)
    
    def convert_to_grayscale(self):
        """Convert the image to grayscale and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        return self.add_filter("Grayscale", 'grayscale')
    
    def resize_image(self, width_percent, height_percent):
        """Resize the image and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {
            'width_percent': width_percent,
            'height_percent': height_percent
        }
        return self.add_filter(f"Resize {width_percent}%x{height_percent}%", 'resize', params)
    
    def crop_image(self, x, y, width, height):
        """Crop the image and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
        return self.add_filter(f"Crop {width}x{height}", 'crop', params)
    
    def flip_image(self, flip_code):
        """Flip the image and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        direction = "Horizontal" if flip_code == 1 else "Vertical"
        params = {'flip_code': flip_code}
        return self.add_filter(f"Flip {direction}", 'flip', params)
    
    def rotate_image(self, angle, scale=1.0):
        """Rotate the image and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {
            'angle': angle,
            'scale': scale
        }
        return self.add_filter(f"Rotate {angle}°", 'rotate', params)
    
    def adjust_brightness_contrast(self, brightness=0, contrast=1.0, preview=False):
        """Adjust brightness and contrast and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {
            'brightness': brightness,
            'contrast': contrast
        }
        return self.add_filter(f"Brightness/Contrast", 'brightness_contrast', params,preview=preview)
    
    def adjust_color_balance(self, r_factor=1.0, g_factor=1.0, b_factor=1.0 ,preview=False):
        """Adjust color balance and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {
            'r_factor': r_factor,
            'g_factor': g_factor,
            'b_factor': b_factor
        }
        return self.add_filter("Color Balance", 'color_balance', params,preview=preview)
    
    def adjust_hue_saturation(self, hue_shift=0, saturation_factor=1.0, preview=False):
        """Adjust hue and saturation and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {
            'hue_shift': hue_shift,
            'saturation_factor': saturation_factor
        }
        return self.add_filter("Hue/Saturation", 'hue_saturation', params,preview=preview)
    
    def apply_gaussian_blur(self, kernel_size=5, preview=False):
        """Apply Gaussian blur and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {'kernel_size': kernel_size}
        return self.add_filter(f"Gaussian Blur (k={kernel_size})", 'gaussian_blur', params,preview=preview)
    
    def reduce_noise(self, strength=10):
        """Reduce noise and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {'strength': strength}
        return self.add_filter(f"Noise Reduction (s={strength})", 'noise_reduction', params)
    
    def equalize_histogram(self):
        """Equalize histogram and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        return self.add_filter("Histogram Equalization", 'histogram_equalization')
    
    def reset_to_original(self):
        """Reset the processed image to the original image and clear filter stack"""
        if self.original_image is None:
            raise ValueError("No original image available")
        
        self.clear_filters()
        self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        return True
    
    def get_histogram_data(self, channel='rgb'):
        """Get histogram data for the current processed image
        
        Args:
            channel: Can be 'rgb', 'r', 'g', 'b', or 'gray'
        
        Returns:
            Dictionary with histogram data
        """
        if self.processed_image is None:
            raise ValueError("No processed image available")
        
        # Convert image to BGR for OpenCV
        if channel == 'gray':
            # Convert to grayscale if needed
            if len(self.processed_image.shape) == 3:
                img = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                img = self.processed_image
            
            # Calculate histogram
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            return {'gray': hist.flatten()}
        else:
            # RGB histogram
            color_data = {}
            colors = {'b': 0, 'g': 1, 'r': 2} if len(self.processed_image.shape) == 3 else {'gray': 0}
            
            for color_name, color_idx in colors.items():
                if channel == 'rgb' or channel == color_name:
                    if len(self.processed_image.shape) == 3:
                        hist = cv2.calcHist([self.processed_image], [color_idx], None, [256], [0, 256])
                        color_data[color_name] = hist.flatten()
                    else:
                        hist = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
                        color_data[color_name] = hist.flatten()
            
            return color_data


class ImageProcessingApp:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Modern Image Processing")
        self.root.geometry("1200x800")
        
        self.style = ttk.Style("darkly")
        self.processor = ImageProcessor()
        
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()
        
        self.current_image_path = None
        self.image_loaded = False
        
        self.preview_mode = False
        self.orig_preview_image = None
        
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_rect_id = None
        self.cropping = False
        
        self.histogram_visible = False
    
    def _create_menu(self):
        """Create the application menu"""
        self.menu_bar = tk.Menu(self.root)
        
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.load_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_image_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Alt+F4")
        
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        edit_menu.add_command(label="Reset to Original", command=self.reset_to_original)
        edit_menu.add_separator()
        
        basic_menu = tk.Menu(edit_menu, tearoff=0)
        basic_menu.add_command(label="Convert to Grayscale", command=self.convert_to_grayscale)
        basic_menu.add_command(label="Flip Horizontally", command=lambda: self.flip_image(1))
        basic_menu.add_command(label="Flip Vertically", command=lambda: self.flip_image(0))
        basic_menu.add_command(label="Rotate", command=self.rotate_image)
        basic_menu.add_command(label="Resize", command=self.resize_image_dialog)
        basic_menu.add_command(label="Crop", command=self.start_crop_mode)
        edit_menu.add_cascade(label="Basic Transformations", menu=basic_menu)
        
        filters_menu = tk.Menu(edit_menu, tearoff=0)
        filters_menu.add_command(label="Brightness & Contrast", command=self.adjust_brightness_contrast_dialog)
        filters_menu.add_command(label="Color Balance", command=self.adjust_color_balance_dialog)
        filters_menu.add_command(label="Hue & Saturation", command=self.adjust_hue_saturation_dialog)
        filters_menu.add_command(label="Gaussian Blur", command=self.apply_gaussian_blur_dialog)
        filters_menu.add_command(label="Noise Reduction", command=self.reduce_noise_dialog)
        filters_menu.add_command(label="Histogram Equalization", command=self.equalize_histogram)
        edit_menu.add_cascade(label="Filters", menu=filters_menu)
        
        advanced_menu = tk.Menu(edit_menu, tearoff=0)
        edge_menu = tk.Menu(advanced_menu, tearoff=0)
        edge_menu.add_command(label="Sobel", command=lambda: self.apply_edge_detection('sobel'))
        edge_menu.add_command(label="Laplacian", command=lambda: self.apply_edge_detection('laplacian'))
        edge_menu.add_command(label="Canny", command=lambda: self.apply_edge_detection('canny', None))
        advanced_menu.add_cascade(label="Edge Detection", menu=edge_menu)
        
        advanced_menu.add_command(label="Shear", command=self.apply_shear_dialog)
        edit_menu.add_cascade(label="Advanced Filters", menu=advanced_menu)
        
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(label="Toggle Preview Mode", command=self.toggle_preview_mode)
        view_menu.add_command(label="Show Histogram", command=self.show_histogram)
        
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        
        self.root.config(menu=self.menu_bar)
        self.root.bind("<Control-o>", lambda event: self.load_image())
        self.root.bind("<Control-s>", lambda event: self.save_image())
        self.root.bind("<Control-Shift-S>", lambda event: self.save_image_as())
    
    def _create_main_layout(self):
        """Create the main application layout"""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)
        
        self.sidebar = ScrolledFrame(self.main_frame, width=250)
        self.sidebar.pack(side=LEFT, fill=Y, padx=(0, 10))
        
        sidebar_label = ttk.Label(self.sidebar, text="Basic Operations", font=("Helvetica", 14))
        sidebar_label.pack(pady=10)
        
        self._create_sidebar_controls()
        
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(side=LEFT, fill=BOTH, expand=YES)
        
        self.filter_stack_frame = ttk.LabelFrame(self.content_frame, text="Applied Filters")
        self.filter_stack_frame.pack(fill=X, pady=(0, 10))
        
        self.filter_list_frame = ScrolledFrame(self.filter_stack_frame, height=100)
        self.filter_list_frame.pack(fill=X, expand=YES, padx=5, pady=5)
        
        self.no_filters_label = ttk.Label(
            self.filter_list_frame, 
            text="No filters applied", 
            font=("Helvetica", 10, "italic")
        )
        self.no_filters_label.pack(pady=10)
        
        self.image_display_frame = ttk.Frame(self.content_frame)
        self.image_display_frame.pack(fill=BOTH, expand=YES)
        
        self.single_view_frame = ttk.Frame(self.image_display_frame)
        self.single_view_frame.pack(fill=BOTH, expand=YES)
        
        self.image_canvas = tk.Canvas(
            self.single_view_frame,
            bg="#2b3e50",
            highlightthickness=0
        )
        self.image_canvas.pack(fill=BOTH, expand=YES)
        
        self.image_canvas.bind("<ButtonPress-1>", self.crop_start)
        self.image_canvas.bind("<B1-Motion>", self.crop_move)
        self.image_canvas.bind("<ButtonRelease-1>", self.crop_end)
        
        self.split_view_frame = ttk.Frame(self.image_display_frame)
        
        self.original_canvas = tk.Canvas(
            self.split_view_frame,
            bg="#2b3e50",
            highlightthickness=0
        )
        
        self.processed_canvas = tk.Canvas(
            self.split_view_frame,
            bg="#2b3e50", 
            highlightthickness=0
        )
        
        self.histogram_frame = ttk.Frame(self.content_frame, height=200)
        
        self.no_image_label = ttk.Label(
            self.image_canvas,
            text="No image loaded\nUse File > Open Image to load an image",
            font=("Helvetica", 14),
            justify="center"
        )
        self.no_image_label.place(relx=0.5, rely=0.5, anchor=CENTER)
    
    def _create_sidebar_controls(self):
        """Create controls in the sidebar"""
        file_frame = ttk.LabelFrame(self.sidebar, text="File Operations")
        file_frame.pack(fill=X, padx=(0,15), pady=5)
        
        load_btn = ttk.Button(
            file_frame,
            text="Load Image",
            command=self.load_image,
            style="primary.TButton"
        )
        load_btn.pack(pady=5, fill=X, padx=5)
        
        save_btn = ttk.Button(
            file_frame,
            text="Save Image",
            command=self.save_image,
            style="success.TButton"
        )
        save_btn.pack(pady=5, fill=X, padx=5)
        
        transform_frame = ttk.LabelFrame(self.sidebar, text="Basic Transformations")
        transform_frame.pack(fill=X, padx=(0,15), pady=5)
        
        grayscale_btn = ttk.Button(
            transform_frame,
            text="Convert to Grayscale",
            command=self.convert_to_grayscale
        )
        grayscale_btn.pack(pady=5, fill=X, padx=5)
        
        resize_btn = ttk.Button(
            transform_frame,
            text="Resize Image",
            command=self.resize_image_dialog
        )
        resize_btn.pack(pady=5, fill=X, padx=5)
        
        crop_btn = ttk.Button(
            transform_frame,
            text="Crop Image",
            command=self.start_crop_mode
        )
        crop_btn.pack(pady=5, fill=X, padx=5)
        
        flip_frame = ttk.Frame(transform_frame)
        flip_frame.pack(fill=X, padx=5, pady=5)
        
        flip_h_btn = ttk.Button(
            flip_frame,
            text="Flip Horizontal",
            command=lambda: self.flip_image(1),
            width=12
        )
        flip_h_btn.pack(side=LEFT, padx=(0, 2), fill=X, expand=YES)
        
        flip_v_btn = ttk.Button(
            flip_frame,
            text="Flip Vertical",
            command=lambda: self.flip_image(0),
            width=12
        )
        flip_v_btn.pack(side=RIGHT, padx=(2, 0), fill=X, expand=YES)
        
        rotation_frame = ttk.Frame(transform_frame)
        rotation_frame.pack(fill=X, padx=(0,0))
        
        rotate_label = ttk.Label(rotation_frame, text="Rotation:")
        rotate_label.pack(side=LEFT,padx=(0,0))
        
        self.rotation_var = tk.IntVar(value=0)
        rotation_slider = ttk.Scale(
            rotation_frame,
            from_=0,
            to=360,
            variable=self.rotation_var,
            orient=HORIZONTAL,
            command=lambda val: self.rotation_var.set(f'{int(float(val)):03}'),
            length=130

        )
        rotation_slider.pack(side=LEFT, fill="none", expand=YES)
        
        rotation_value = ttk.Label(rotation_frame, textvariable=self.rotation_var)
        rotation_value.pack(side=RIGHT)
        
        rotate_btn = ttk.Button(
            transform_frame,
            text="Apply Rotation",
            command=lambda: self.rotate_image(angle=self.rotation_var.get())
        )
        rotate_btn.pack(pady=5, fill=X, padx=5)
        
        filters_frame = ttk.LabelFrame(self.sidebar, text="Filters")
        filters_frame.pack(fill=X, padx=(0,15), pady=5)
        
        bright_contrast_btn = ttk.Button(
            filters_frame,
            text="Brightness & Contrast",
            command=self.adjust_brightness_contrast_dialog
        )
        bright_contrast_btn.pack(pady=5, fill=X, padx=5)
        
        color_btn = ttk.Button(
            filters_frame,
            text="Color Balance",
            command=self.adjust_color_balance_dialog
        )
        color_btn.pack(pady=5, fill=X, padx=5)
        
        hue_sat_btn = ttk.Button(
            filters_frame,
            text="Hue & Saturation",
            command=self.adjust_hue_saturation_dialog
        )
        hue_sat_btn.pack(pady=5, fill=X, padx=5)
        
        blur_btn = ttk.Button(
            filters_frame,
            text="Gaussian Blur",
            command=self.apply_gaussian_blur_dialog
        )
        blur_btn.pack(pady=5, fill=X, padx=5)
        
        hist_eq_btn = ttk.Button(
            filters_frame,
            text="Histogram Equalization",
            command=self.equalize_histogram
        )
        hist_eq_btn.pack(pady=5, fill=X, padx=5)
        
        advanced_frame = ttk.LabelFrame(self.sidebar, text="Advanced Filters")
        advanced_frame.pack(fill=X, padx=(0,15), pady=5)
        
        edge_label = ttk.Label(advanced_frame, text="Edge Detection:")
        edge_label.pack(anchor=W, padx=5, pady=(5, 0))
        
        edge_frame = ttk.Frame(advanced_frame)
        edge_frame.pack(fill=X, padx=5, pady=5)
        
        sobel_btn = ttk.Button(
            edge_frame,
            text="Sobel",
            command=lambda: self.apply_edge_detection('sobel'),
            width=8
        )
        sobel_btn.pack(side=LEFT, padx=(0, 2), fill=X, expand=YES)
        
        laplacian_btn = ttk.Button(
            edge_frame,
            text="Laplacian",
            command=lambda: self.apply_edge_detection('laplacian'),
            width=8
        )
        laplacian_btn.pack(side=LEFT, padx=2, fill=X, expand=YES)
        
        canny_btn = ttk.Button(
            edge_frame,
            text="Canny",
            command=lambda: self.apply_edge_detection('canny', None),
            width=8
        )
        canny_btn.pack(side=LEFT, padx=(2, 0), fill=X, expand=YES)
        
        shear_btn = ttk.Button(
            advanced_frame,
            text="Shear Transform",
            command=self.apply_shear_dialog
        )
        shear_btn.pack(pady=5, fill=X, padx=5)
        
        view_frame = ttk.LabelFrame(self.sidebar, text="View Options")
        view_frame.pack(fill=X, padx=(0,15), pady=5)
        
        preview_btn = ttk.Button(
            view_frame,
            text="Toggle Preview Mode",
            command=self.toggle_preview_mode
        )
        preview_btn.pack(pady=5, fill=X, padx=5)
        
        histogram_btn = ttk.Button(
            view_frame,
            text="Show Histogram",
            command=self.show_histogram
        )
        histogram_btn.pack(pady=5, fill=X, padx=5)
        
        reset_btn = ttk.Button(
            view_frame,
            text="Reset to Original",
            command=self.reset_to_original,
            style="warning.TButton"
        )
        reset_btn.pack(pady=5, fill=X, padx=5)
    
    def _create_status_bar(self):
        """Create the status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(10, 5)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def update_filter_stack_display(self):
        """Update the filter stack display"""
        for widget in self.filter_list_frame.winfo_children():
            widget.destroy()
        
        filters = self.processor.filter_stack.get_filters()
        
        if not filters:
            self.no_filters_label = ttk.Label(
                self.filter_list_frame, 
                text="No filters applied", 
                font=("Helvetica", 10, "italic")
            )
            self.no_filters_label.pack(pady=10)
            self.show_histogram(toggle=False)
            return
        
        for i, filter_obj in enumerate(filters):
            filter_frame = ttk.Frame(self.filter_list_frame)
            filter_frame.pack(fill=X, padx=5, pady=2)
            
            filter_name = ttk.Label(
                filter_frame, 
                text=f"{i+1}. {filter_obj.name}", 
                width=20,
                anchor=W
            )
            filter_name.pack(side=LEFT, padx=(0, 5))
            
            delete_btn = ttk.Button(
                filter_frame,
                text="×",
                command=lambda fid=filter_obj.id: self.remove_filter(fid),
                width=2,
                style="danger.TButton"
            )
            delete_btn.pack(side=RIGHT, padx=(5, 0))
            
            up_btn = ttk.Button(
                filter_frame,
                text="↑",
                command=lambda fid=filter_obj.id, pos=i: self.move_filter_up(fid, pos),
                width=2,
                state=DISABLED if i == 0 else NORMAL
            )
            up_btn.pack(side=RIGHT, padx=1)
            
            down_btn = ttk.Button(
                filter_frame,
                text="↓",
                command=lambda fid=filter_obj.id, pos=i: self.move_filter_down(fid, pos),
                width=2,
                state=DISABLED if i == len(filters) - 1 else NORMAL
            )
            down_btn.pack(side=RIGHT, padx=1)
        
        # Refresh histogram if it's currently visible
        self.show_histogram(toggle=False)
    
    def remove_filter(self, filter_id):
        """Remove a filter from the stack"""
        try:
            self.update_status("Removing filter...")
            if self.processor.remove_filter(filter_id):
                self.update_filter_stack_display()
                self.display_image()
                self.update_status("Filter removed")
            else:
                self.update_status("Failed to remove filter")
        except Exception as e:
            self.handle_error(f"Error removing filter: {str(e)}")
    
    def move_filter_up(self, filter_id, current_position):
        """Move a filter up in the stack"""
        try:
            self.update_status("Reordering filters...")
            if self.processor.move_filter(filter_id, current_position - 1):
                self.update_filter_stack_display()
                self.display_image()
                self.update_status("Filter reordered")
            else:
                self.update_status("Failed to reorder filter")
        except Exception as e:
            self.handle_error(f"Error reordering filter: {str(e)}")
    
    def move_filter_down(self, filter_id, current_position):
        """Move a filter down in the stack"""
        try:
            self.update_status("Reordering filters...")
            if self.processor.move_filter(filter_id, current_position + 1):
                self.update_filter_stack_display()
                self.display_image()
                self.update_status("Filter reordered")
            else:
                self.update_status("Failed to reorder filter")
        except Exception as e:
            self.handle_error(f"Error reordering filter: {str(e)}")
    
    def display_image(self):
        """Display the current image on the canvas"""
        if not self.image_loaded:
            self.no_image_label.place(relx=0.5, rely=0.5, anchor=CENTER)
            return
        
        self.no_image_label.place_forget()
        
        if self.preview_mode:
            self.single_view_frame.pack_forget()
            self.split_view_frame.pack(fill=BOTH, expand=YES)
            
            if not self.original_canvas.winfo_ismapped():
                self.original_canvas.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 2))
                self.processed_canvas.pack(side=RIGHT, fill=BOTH, expand=YES, padx=(2, 0))
            
            orig_width = self.original_canvas.winfo_width()
            orig_height = self.original_canvas.winfo_height()
            
            if self.orig_preview_image is None:
                original_rgb = cv2.cvtColor(self.processor.original_image, cv2.COLOR_BGR2RGB)
                self.orig_preview_image = original_rgb
            
            pil_orig = self.processor.get_display_image(width=orig_width, height=orig_height, 
                                                      image=self.orig_preview_image)
            if pil_orig:
                self.tk_orig_image = ImageTk.PhotoImage(pil_orig)
                self.original_canvas.delete("all")
                self.original_canvas.create_image(
                    orig_width // 2,
                    orig_height // 2,
                    anchor=CENTER,
                    image=self.tk_orig_image
                )
                self.original_canvas.create_text(
                    10, 10, 
                    text="Original", 
                    fill="white", 
                    anchor=NW,
                    font=("Helvetica", 10, "bold")
                )
            
            proc_width = self.processed_canvas.winfo_width()
            proc_height = self.processed_canvas.winfo_height()
            
            pil_proc = self.processor.get_display_image(width=proc_width, height=proc_height)
            if pil_proc:
                self.tk_proc_image = ImageTk.PhotoImage(pil_proc)
                self.processed_canvas.delete("all")
                self.processed_canvas.create_image(
                    proc_width // 2,
                    proc_height // 2,
                    anchor=CENTER,
                    image=self.tk_proc_image
                )
                self.processed_canvas.create_text(
                    10, 10, 
                    text="Processed", 
                    fill="white", 
                    anchor=NW,
                    font=("Helvetica", 10, "bold")
                )
        else:
            self.split_view_frame.pack_forget()
            self.single_view_frame.pack(fill=BOTH, expand=YES)
            
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            pil_image = self.processor.get_display_image(width=canvas_width, height=canvas_height)
            if pil_image:
                self.tk_image = ImageTk.PhotoImage(pil_image)
                self.image_canvas.delete("all")
                self.image_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor=CENTER,
                    image=self.tk_image
                )
        
        h, w = self.processor.processed_image.shape[:2]
        self.update_status(f"Image: {w}x{h} pixels")
    
    def load_image(self, event=None):
        """Load an image from file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Open Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("PNG", "*.png"),
                    ("BMP", "*.bmp"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return
            
            self.update_status(f"Loading image: {file_path}")
            
            if self.processor.load_image(file_path):
                self.current_image_path = file_path
                self.image_loaded = True
                
                self.processor.clear_filters()
                self.update_filter_stack_display()
                
                self.orig_preview_image = None
                
                self.root.after(100, self.display_image)
                
                self.update_status(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            self.handle_error(f"Error loading image: {str(e)}")
    
    def save_image(self, event=None):
        """Save the current image"""
        if not self.image_loaded:
            self.handle_error("No image to save")
            return
        
        try:
            if self.current_image_path:
                self.processor.save_image(self.current_image_path)
                self.update_status(f"Saved to: {os.path.basename(self.current_image_path)}")
            else:
                self.save_image_as()
                
        except Exception as e:
            self.handle_error(f"Error saving image: {str(e)}")
    
    def save_image_as(self, event=None):
        """Save the current image with a new filename"""
        if not self.image_loaded:
            self.handle_error("No image to save")
            return
        
        try:
            default_ext = ""
            if self.current_image_path:
                _, ext = os.path.splitext(self.current_image_path)
                default_ext = ext if ext else ".png"
            
            file_path = filedialog.asksaveasfilename(
                title="Save Image As",
                defaultextension=default_ext if default_ext else ".png",
                filetypes=[
                    ("JPEG", "*.jpg"),
                    ("PNG", "*.png"),
                    ("BMP", "*.bmp"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return
            
            self.update_status(f"Saving image to: {file_path}")
            
            if self.processor.save_image(file_path):
                self.current_image_path = file_path
                self.update_status(f"Saved to: {os.path.basename(file_path)}")
                
        except Exception as e:
            self.handle_error(f"Error saving image: {str(e)}")
    
    def reset_to_original(self):
        """Reset the processed image to the original image"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            self.update_status("Resetting to original image...")
            self.processor.reset_to_original()
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Reset to original image")
            self.show_histogram(toggle=False)
        except Exception as e:
            self.handle_error(f"Error resetting image: {str(e)}")
    
    def handle_error(self, message):
        """Handle and display errors"""
        self.update_status(f"Error: {message}")
        messagebox.showerror("Error", message)
    
    def on_resize(self, event=None):
        """Handle window resize events"""
        if self.image_loaded:
            self.root.after_cancel(self.after_id) if hasattr(self, 'after_id') else None
            self.after_id = self.root.after(100, self.display_image)
    
    def toggle_preview_mode(self):
        """Toggle between single view and side-by-side preview mode"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        self.preview_mode = not self.preview_mode
        
        if self.preview_mode:
            self.update_status("Preview mode: Showing original and processed images")
        else:
            self.update_status("Preview mode: Showing processed image only")
        
        self.display_image()
    
    def show_histogram(self, toggle=True):
        """Display histogram for the current image"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            if toggle:
                self.histogram_visible = not self.histogram_visible
            
            if not self.histogram_visible:
                self.histogram_frame.pack_forget()
                self.update_status("Histogram hidden")
                return
            
            self.update_status("Generating histogram...")
            
            hist_data = self.processor.get_histogram_data()
            
            for widget in self.histogram_frame.winfo_children():
                widget.destroy()
            
            fig = Figure(figsize=(5, 2), dpi=100)
            fig.patch.set_facecolor('#2b3e50')
            
            ax = fig.add_subplot(111)
            ax.set_facecolor('#2b3e50')
            
            if 'gray' in hist_data:
                ax.plot(hist_data['gray'], color='white')
                ax.set_title('Grayscale Histogram', color='white')
            else:
                ax.plot(hist_data['r'], color='red', label='Red')
                ax.plot(hist_data['g'], color='green', label='Green')
                ax.plot(hist_data['b'], color='blue', label='Blue')
                ax.legend()
                ax.set_title('RGB Histogram', color='white')
            
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=YES)
            
            self.histogram_frame.pack_forget()
            self.image_display_frame.pack_forget()
            
            self.image_display_frame.pack(fill=BOTH, expand=YES)
            self.histogram_frame.pack(fill=X, pady=(10, 0))
            
            self.update_status("Histogram displayed")
            
        except Exception as e:
            self.handle_error(f"Error displaying histogram: {str(e)}")
    
    def convert_to_grayscale(self):
        """Convert image to grayscale"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            self.update_status("Converting to grayscale...")
            self.processor.convert_to_grayscale()
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Converted to grayscale")
        except Exception as e:
            self.handle_error(f"Error converting to grayscale: {str(e)}")
    
    def resize_image_dialog(self):
        """Show dialog to resize image"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            resize_dialog = ttk.Toplevel(self.root)
            resize_dialog.title("Resize Image")
            resize_dialog.geometry("300x200")
            resize_dialog.transient(self.root)
            resize_dialog.grab_set()
            
            controls_frame = ttk.Frame(resize_dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            width_frame = ttk.Frame(controls_frame)
            width_frame.pack(fill=X, pady=5)
            
            width_label = ttk.Label(width_frame, text="Width %:")
            width_label.pack(side=LEFT)
            
            width_var = tk.IntVar(value=100)
            width_slider = ttk.Scale(
                width_frame,
                from_=10,
                to=200,
                variable=width_var,
                command=lambda val: width_var.set(int(float(val))),
                orient=HORIZONTAL
            )
            width_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            width_value = ttk.Label(width_frame, textvariable=width_var)
            width_value.pack(side=LEFT, padx=(0, 5))
            
            height_frame = ttk.Frame(controls_frame)
            height_frame.pack(fill=X, pady=5)
            
            height_label = ttk.Label(height_frame, text="Height %:")
            height_label.pack(side=LEFT)
            
            height_var = tk.IntVar(value=100)
            height_slider = ttk.Scale(
                height_frame,
                from_=10,
                to=200,
                variable=height_var,
                command=lambda val: height_var.set(int(float(val))),
                orient=HORIZONTAL
            )
            height_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            height_value = ttk.Label(height_frame, textvariable=height_var)
            height_value.pack(side=LEFT, padx=(0, 5))
            
            lock_var = tk.BooleanVar(value=True)
            lock_check = ttk.Checkbutton(
                controls_frame, 
                text="Lock aspect ratio",
                variable=lock_var
            )
            lock_check.pack(anchor=W, pady=5)
            
            def update_height(*args):
                if lock_var.get():
                    height_var.set(width_var.get())
            
            def update_width(*args):
                if lock_var.get():
                    width_var.set(height_var.get())
            
            width_var.trace_add("write", update_height)
            height_var.trace_add("write", update_width)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            cancel_btn = ttk.Button(
                button_frame,
                text="Cancel",
                command=resize_dialog.destroy
            )
            cancel_btn.pack(side=RIGHT, padx=5)
            
            apply_btn = ttk.Button(
                button_frame,
                text="Apply",
                style="primary.TButton",
                command=lambda: self.apply_resize(width_var.get(), height_var.get(), resize_dialog)
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.handle_error(f"Error creating resize dialog: {str(e)}")
    
    def apply_resize(self, width_percent, height_percent, dialog=None):
        """Apply resize operation with the specified percentages"""
        try:
            self.update_status(f"Resizing image to {width_percent}% width, {height_percent}% height...")
            self.processor.resize_image(width_percent, height_percent)
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Image resized")
            
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            self.handle_error(f"Error resizing image: {str(e)}")
    
    def start_crop_mode(self):
        """Start crop mode to allow selecting a crop area"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            if self.preview_mode:
                self.toggle_preview_mode()
            
            self.cropping = True
            self.update_status("Click and drag to select crop area. Press ESC to cancel.")
            
            self.image_canvas.config(cursor="crosshair")
            
        except Exception as e:
            self.handle_error(f"Error starting crop mode: {str(e)}")
    
    def crop_start(self, event):
        """Handle mouse press for cropping"""
        if not self.cropping:
            return
        
        self.crop_start_x = self.image_canvas.canvasx(event.x)
        self.crop_start_y = self.image_canvas.canvasy(event.y)
        
        if self.crop_rect_id:
            self.image_canvas.delete(self.crop_rect_id)
        
        self.crop_rect_id = self.image_canvas.create_rectangle(
            self.crop_start_x, 
            self.crop_start_y, 
            self.crop_start_x, 
            self.crop_start_y,
            outline="lime",
            width=2
        )
    
    def crop_move(self, event):
        """Handle mouse drag for cropping"""
        if not self.cropping or self.crop_rect_id is None:
            return
        
        cur_x = self.image_canvas.canvasx(event.x)
        cur_y = self.image_canvas.canvasy(event.y)
        
        self.image_canvas.coords(
            self.crop_rect_id,
            self.crop_start_x,
            self.crop_start_y,
            cur_x,
            cur_y
        )
    
    def crop_end(self, event):
        """Handle mouse release for cropping"""
        if not self.cropping or self.crop_rect_id is None:
            return
        
        end_x = self.image_canvas.canvasx(event.x)
        end_y = self.image_canvas.canvasy(event.y)
        
        self.cropping = False
        self.image_canvas.config(cursor="")
        
        image_item = self.image_canvas.find_withtag("all")[0]
        img_bbox = self.image_canvas.bbox(image_item)
        
        if img_bbox:
            img_width = img_bbox[2] - img_bbox[0]
            img_height = img_bbox[3] - img_bbox[1]
            
            img_real_width = self.processor.processed_image.shape[1]
            img_real_height = self.processor.processed_image.shape[0]
            
            scale_x = img_real_width / img_width
            scale_y = img_real_height / img_height
            
            canvas_x1 = min(self.crop_start_x, end_x) - img_bbox[0]
            canvas_y1 = min(self.crop_start_y, end_y) - img_bbox[1]
            canvas_x2 = max(self.crop_start_x, end_x) - img_bbox[0]
            canvas_y2 = max(self.crop_start_y, end_y) - img_bbox[1]
            
            img_x1 = max(0, int(canvas_x1 * scale_x))
            img_y1 = max(0, int(canvas_y1 * scale_y))
            img_x2 = min(img_real_width, int(canvas_x2 * scale_x))
            img_y2 = min(img_real_height, int(canvas_y2 * scale_y))
            
            if img_x2 - img_x1 > 10 and img_y2 - img_y1 > 10:
                self.apply_crop(img_x1, img_y1, img_x2 - img_x1, img_y2 - img_y1)
            else:
                self.update_status("Crop area too small. Selection cancelled.")
        
        if self.crop_rect_id:
            self.image_canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
    
    def apply_crop(self, x, y, width, height):
        """Apply crop with the selected coordinates"""
        try:
            self.update_status(f"Cropping image to {width}x{height} at ({x},{y})...")
            self.processor.crop_image(x, y, width, height)
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Image cropped")
        except Exception as e:
            self.handle_error(f"Error cropping image: {str(e)}")
    
    def flip_image(self, flip_code):
        """Flip the image horizontally or vertically"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            flip_type = "horizontally" if flip_code == 1 else "vertically"
            self.update_status(f"Flipping image {flip_type}...")
            self.processor.flip_image(flip_code)
            self.update_filter_stack_display()
            self.display_image()
            self.update_status(f"Image flipped {flip_type}")
        except Exception as e:
            self.handle_error(f"Error flipping image: {str(e)}")
    
    def rotate_image(self, angle=None):
        """Rotate the image by the specified angle"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            if angle is None:
                angle = simpledialog.askfloat(
                    "Rotate Image",
                    "Enter rotation angle (degrees):",
                    initialvalue=0,
                    minvalue=0,
                    maxvalue=360
                )
                
                if angle is None:
                    return
            
            self.update_status(f"Rotating image by {angle} degrees...")
            self.processor.rotate_image(angle)
            self.update_filter_stack_display()
            self.display_image()
            self.update_status(f"Image rotated by {angle} degrees")
        except Exception as e:
            self.handle_error(f"Error rotating image: {str(e)}")
    
    def adjust_brightness_contrast_dialog(self):
        """Show dialog to adjust brightness and contrast"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            dialog = ttk.Toplevel(self.root)
            dialog.title("Adjust Brightness & Contrast")
            dialog.geometry("350x200")
            dialog.transient(self.root)
            dialog.grab_set()
            
            controls_frame = ttk.Frame(dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            brightness_frame = ttk.Frame(controls_frame)
            brightness_frame.pack(fill=X, pady=5)
            
            brightness_label = ttk.Label(brightness_frame, text="Brightness:")
            brightness_label.pack(side=LEFT)
            
            brightness_var = tk.IntVar(value=0)
            brightness_slider = ttk.Scale(
                brightness_frame,
                from_=-100,
                to=100,
                variable=brightness_var,
                orient=HORIZONTAL
            )
            brightness_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)

            formatted_brightness = tk.StringVar()

            def update_brightness_text(*args):
                formatted_brightness.set(f"{brightness_var.get():.2f}")

            brightness_var.trace_add("write", update_brightness_text)
            
            brightness_value = ttk.Label(brightness_frame, textvariable=formatted_brightness)
            brightness_value.pack(side=LEFT, padx=(0, 5))
            
            contrast_frame = ttk.Frame(controls_frame)
            contrast_frame.pack(fill=X, pady=5)
            
            contrast_label = ttk.Label(contrast_frame, text="Contrast:")
            contrast_label.pack(side=LEFT)
            
            contrast_var = tk.DoubleVar(value=1.0)
            contrast_slider = ttk.Scale(
                contrast_frame,
                from_=0.5,
                to=2.0,
                variable=contrast_var,
                orient=HORIZONTAL,
                command= lambda v: contrast_var.set(float(v).__round__(2))
            )
            contrast_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            formatted_contrast = tk.StringVar()
            
            def update_contrast_text(*args):
                formatted_contrast.set(f"{contrast_var.get():.2f}")
            
            contrast_var.trace_add("write", update_contrast_text)
            update_contrast_text()
            
            contrast_value = ttk.Label(contrast_frame, textvariable=formatted_contrast)
            contrast_value.pack(side=LEFT, padx=(0, 5))
            
            preview_var = tk.BooleanVar(value=False)
            preview_check = ttk.Checkbutton(
                controls_frame, 
                text="Live Preview",
                variable=preview_var
            )
            preview_check.pack(anchor=W, pady=5)
            
            original_img = self.processor.processed_image.copy()
            
            def update_preview(*args):
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.processor.adjust_brightness_contrast(
                        brightness=brightness_var.get(),
                        contrast=contrast_var.get(),
                        preview=True
                    )
                    self.display_image()
            
            brightness_var.trace_add("write", update_preview)
            contrast_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.display_image()
                dialog.destroy()
            
            cancel_btn = ttk.Button(
                button_frame,
                text="Cancel",
                command=on_cancel
            )
            cancel_btn.pack(side=RIGHT, padx=5)
            
            apply_btn = ttk.Button(
                button_frame,
                text="Apply",
                style="primary.TButton",
                command=lambda: self.apply_brightness_contrast(
                    brightness_var.get(), 
                    contrast_var.get(), 
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.handle_error(f"Error creating brightness/contrast dialog: {str(e)}")
    
    def apply_brightness_contrast(self, brightness, contrast, dialog=None):
        """Apply brightness and contrast adjustments to the image"""
        try:
            self.update_status(f"Adjusting brightness ({brightness}) and contrast ({contrast:.2f})...")
            self.processor.adjust_brightness_contrast(brightness=brightness, contrast=contrast)
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Brightness and contrast adjusted")
            
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            self.handle_error(f"Error adjusting brightness/contrast: {str(e)}")
    
    def adjust_color_balance_dialog(self):
        """Show dialog to adjust color balance"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            dialog = ttk.Toplevel(self.root)
            dialog.title("Color Balance")
            dialog.geometry("350x250")
            dialog.transient(self.root)
            dialog.grab_set()
            
            controls_frame = ttk.Frame(dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            red_frame = ttk.Frame(controls_frame)
            red_frame.pack(fill=X, pady=5)
            
            red_label = ttk.Label(red_frame, text="Red:", foreground="red")
            red_label.pack(side=LEFT)
            
            r_factor_var = tk.DoubleVar(value=1.0)
            red_slider = ttk.Scale(
                red_frame,
                from_=0.0,
                to=2.0,
                variable=r_factor_var,
                command=lambda val: r_factor_var.set(float(val).__round__(2)),
                orient=HORIZONTAL
            )
            red_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            formatted_r = tk.StringVar()
            
            def update_r_text(*args):
                formatted_r.set(f"{r_factor_var.get():.2f}")
            
            r_factor_var.trace_add("write", update_r_text)
            update_r_text()
            
            red_value = ttk.Label(red_frame, textvariable=formatted_r)
            red_value.pack(side=LEFT, padx=(0, 5))
            
            green_frame = ttk.Frame(controls_frame)
            green_frame.pack(fill=X, pady=5)
            
            green_label = ttk.Label(green_frame, text="Green:", foreground="green")
            green_label.pack(side=LEFT)
            
            g_factor_var = tk.DoubleVar(value=1.0)
            green_slider = ttk.Scale(
                green_frame,
                from_=0.0,
                to=2.0,
                variable=g_factor_var,
                command=lambda val: g_factor_var.set(float(val).__round__(2)),
                orient=HORIZONTAL
            )
            green_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            formatted_g = tk.StringVar()
            
            def update_g_text(*args):
                formatted_g.set(f"{g_factor_var.get():.2f}")
            
            g_factor_var.trace_add("write", update_g_text)
            update_g_text()
            
            green_value = ttk.Label(green_frame, textvariable=formatted_g)
            green_value.pack(side=LEFT, padx=(0, 5))
            
            blue_frame = ttk.Frame(controls_frame)
            blue_frame.pack(fill=X, pady=5)
            
            blue_label = ttk.Label(blue_frame, text="Blue:", foreground="blue")
            blue_label.pack(side=LEFT)
            
            b_factor_var = tk.DoubleVar(value=1.0)
            blue_slider = ttk.Scale(
                blue_frame,
                from_=0.0,
                to=2.0,
                variable=b_factor_var,
                command=lambda val: b_factor_var.set(float(val).__round__(2)),
                orient=HORIZONTAL
            )
            blue_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            formatted_b = tk.StringVar()
            
            def update_b_text(*args):
                formatted_b.set(f"{b_factor_var.get():.2f}")
            
            b_factor_var.trace_add("write", update_b_text)
            update_b_text()
            
            blue_value = ttk.Label(blue_frame, textvariable=formatted_b)
            blue_value.pack(side=LEFT, padx=(0, 5))
            
            preview_var = tk.BooleanVar(value=False)
            preview_check = ttk.Checkbutton(
                controls_frame, 
                text="Live Preview",
                variable=preview_var
            )
            preview_check.pack(anchor=W, pady=5)
            
            original_img = self.processor.processed_image.copy()
            
            def update_preview(*args):
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.processor.adjust_color_balance(
                        r_factor=r_factor_var.get(),
                        g_factor=g_factor_var.get(),
                        b_factor=b_factor_var.get(),
                        preview=True
                    )
                    self.display_image()
            
            r_factor_var.trace_add("write", update_preview)
            g_factor_var.trace_add("write", update_preview)
            b_factor_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.display_image()
                dialog.destroy()
            
            cancel_btn = ttk.Button(
                button_frame,
                text="Cancel",
                command=on_cancel
            )
            cancel_btn.pack(side=RIGHT, padx=5)
            
            apply_btn = ttk.Button(
                button_frame,
                text="Apply",
                style="primary.TButton",
                command=lambda: self.apply_color_balance(
                    r_factor_var.get(),
                    g_factor_var.get(),
                    b_factor_var.get(),
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.handle_error(f"Error creating color balance dialog: {str(e)}")
    
    def apply_color_balance(self, r_factor, g_factor, b_factor, dialog=None):
        """Apply color balance adjustments to the image"""
        try:
            self.update_status(f"Adjusting color balance (R:{r_factor:.2f}, G:{g_factor:.2f}, B:{b_factor:.2f})...")
            self.processor.adjust_color_balance(
                r_factor=r_factor,
                g_factor=g_factor,
                b_factor=b_factor
            )
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Color balance adjusted")
            
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            self.handle_error(f"Error adjusting color balance: {str(e)}")
    
    def adjust_hue_saturation_dialog(self):
        """Show dialog to adjust hue and saturation"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            dialog = ttk.Toplevel(self.root)
            dialog.title("Hue & Saturation")
            dialog.geometry("350x200")
            dialog.transient(self.root)
            dialog.grab_set()
            
            controls_frame = ttk.Frame(dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            hue_frame = ttk.Frame(controls_frame)
            hue_frame.pack(fill=X, pady=5)
            
            hue_label = ttk.Label(hue_frame, text="Hue:")
            hue_label.pack(side=LEFT)
            
            hue_var = tk.DoubleVar(value=0.0)
            hue_slider = ttk.Scale(
                hue_frame,
                from_=-30.0,
                to=30.0,
                variable=hue_var,
                orient=HORIZONTAL
            )
            hue_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            formatted_hue = tk.StringVar()

            def update_hue_text(*args):
                formatted_hue.set(f"{hue_var.get():.2f}")

            hue_var.trace_add("write", update_hue_text)

            hue_value = ttk.Label(hue_frame, textvariable=formatted_hue)
            hue_value.pack(side=LEFT, padx=(0, 5))
            
            saturation_frame = ttk.Frame(controls_frame)
            saturation_frame.pack(fill=X, pady=5)
            
            sat_label = ttk.Label(saturation_frame, text="Saturation:")
            sat_label.pack(side=LEFT)
            
            sat_var = tk.DoubleVar(value=1.0)
            sat_slider = ttk.Scale(
                saturation_frame,
                from_=0.0,
                to=2.0,
                variable=sat_var,
                orient=HORIZONTAL
            )
            sat_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            formatted_sat = tk.StringVar()
            
            def update_sat_text(*args):
                formatted_sat.set(f"{sat_var.get():.2f}")
            
            sat_var.trace_add("write", update_sat_text)
            update_sat_text()
            
            sat_value = ttk.Label(saturation_frame, textvariable=formatted_sat)
            sat_value.pack(side=LEFT, padx=(0, 5))
            
            preview_var = tk.BooleanVar(value=False)
            preview_check = ttk.Checkbutton(
                controls_frame, 
                text="Live Preview",
                variable=preview_var
            )
            preview_check.pack(anchor=W, pady=5)
            
            original_img = self.processor.processed_image.copy()
            
            def update_preview(*args):
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.processor.adjust_hue_saturation(
                        hue_shift=hue_var.get(),
                        saturation_factor=sat_var.get(),
                        preview=True
                    )
                    self.display_image()
            
            hue_var.trace_add("write", update_preview)
            sat_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.display_image()
                dialog.destroy()
            
            cancel_btn = ttk.Button(
                button_frame,
                text="Cancel",
                command=on_cancel
            )
            cancel_btn.pack(side=RIGHT, padx=5)
            
            apply_btn = ttk.Button(
                button_frame,
                text="Apply",
                style="primary.TButton",
                command=lambda: self.apply_hue_saturation(
                    hue_var.get(), 
                    sat_var.get(), 
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.handle_error(f"Error creating hue/saturation dialog: {str(e)}")
    
    def apply_hue_saturation(self, hue_shift, saturation_factor, dialog=None):
        """Apply hue and saturation adjustments to the image"""
        try:
            self.update_status(f"Adjusting hue ({hue_shift}) and saturation ({saturation_factor:.2f})...")
            self.processor.adjust_hue_saturation(
                hue_shift=hue_shift,
                saturation_factor=saturation_factor
            )
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Hue and saturation adjusted")
            
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            self.handle_error(f"Error adjusting hue/saturation: {str(e)}")
    
    def apply_gaussian_blur_dialog(self):
        """Show dialog to apply Gaussian blur"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            dialog = ttk.Toplevel(self.root)
            dialog.title("Gaussian Blur")
            dialog.geometry("350x150")
            dialog.transient(self.root)
            dialog.grab_set()
            
            controls_frame = ttk.Frame(dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            kernel_frame = ttk.Frame(controls_frame)
            kernel_frame.pack(fill=X, pady=5)
            
            kernel_label = ttk.Label(kernel_frame, text="Blur Amount:")
            kernel_label.pack(side=LEFT)
            
            kernel_var = tk.IntVar(value=5)
            kernel_slider = ttk.Scale(
                kernel_frame,
                from_=1,
                to=100,
                variable=kernel_var,
                command=lambda val: kernel_var.set(int(float(val))),
                orient=HORIZONTAL
            )
            kernel_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)

            formatted_kernel = tk.StringVar()

            def update_kernel_text(*args):
                formatted_kernel.set(f"{kernel_var.get():.0f}")

            kernel_var.trace_add("write", update_kernel_text)
            
            kernel_value = ttk.Label(kernel_frame, textvariable=formatted_kernel)
            kernel_value.pack(side=LEFT, padx=(0, 5))
            
            preview_var = tk.BooleanVar(value=False)
            preview_check = ttk.Checkbutton(
                controls_frame, 
                text="Live Preview",
                variable=preview_var
            )
            preview_check.pack(anchor=W, pady=5)
            
            original_img = self.processor.processed_image.copy()
            
            def update_preview(*args):
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    kernel_size = kernel_var.get()
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    self.processor.apply_gaussian_blur(kernel_size=kernel_size, preview=True)
                    self.display_image()
            
            kernel_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.display_image()
                dialog.destroy()
            
            cancel_btn = ttk.Button(
                button_frame,
                text="Cancel",
                command=on_cancel
            )
            cancel_btn.pack(side=RIGHT, padx=5)
            
            apply_btn = ttk.Button(
                button_frame,
                text="Apply",
                style="primary.TButton",
                command=lambda: self.apply_gaussian_blur(
                    kernel_var.get() if kernel_var.get() % 2 == 1 else kernel_var.get() + 1, 
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.handle_error(f"Error creating Gaussian blur dialog: {str(e)}")
    
    def apply_gaussian_blur(self, kernel_size, dialog=None):
        """Apply Gaussian blur to the image"""
        try:
            self.update_status(f"Applying Gaussian blur (kernel size: {kernel_size})...")
            self.processor.apply_gaussian_blur(kernel_size=kernel_size)
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Gaussian blur applied")
            
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            self.handle_error(f"Error applying Gaussian blur: {str(e)}")
    
    def reduce_noise_dialog(self):
        """Show dialog to reduce noise"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            dialog = ttk.Toplevel(self.root)
            dialog.title("Noise Reduction")
            dialog.geometry("350x150")
            dialog.transient(self.root)
            dialog.grab_set()
            
            controls_frame = ttk.Frame(dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            strength_frame = ttk.Frame(controls_frame)
            strength_frame.pack(fill=X, pady=5)
            
            strength_label = ttk.Label(strength_frame, text="Strength:")
            strength_label.pack(side=LEFT)
            
            strength_var = tk.IntVar(value=10)
            strength_slider = ttk.Scale(
                strength_frame,
                from_=5,
                to=30,
                variable=strength_var,
                command=lambda val: strength_var.set(int(float(val))),
                orient=HORIZONTAL
            )
            strength_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            strength_value = ttk.Label(strength_frame, textvariable=strength_var)
            strength_value.pack(side=LEFT, padx=(0, 5))
            
            note_label = ttk.Label(
                controls_frame, 
                text="Note: Noise reduction may take some time to process.",
                font=("Helvetica", 9, "italic")
            )
            note_label.pack(pady=5)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            cancel_btn = ttk.Button(
                button_frame,
                text="Cancel",
                command=dialog.destroy
            )
            cancel_btn.pack(side=RIGHT, padx=5)
            
            apply_btn = ttk.Button(
                button_frame,
                text="Apply",
                style="primary.TButton",
                command=lambda: self.apply_noise_reduction(
                    strength_var.get(), 
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.handle_error(f"Error creating noise reduction dialog: {str(e)}")
    
    def apply_noise_reduction(self, strength, dialog=None):
        """Apply noise reduction to the image"""
        try:
            self.update_status(f"Reducing noise (strength: {strength})...")
            
            if dialog:
                dialog.withdraw()
            
            loading = ttk.Toplevel(self.root)
            loading.title("Processing")
            loading.geometry("250x100")
            loading.transient(self.root)
            
            loading_frame = ttk.Frame(loading, padding=20)
            loading_frame.pack(fill=BOTH, expand=YES)
            
            ttk.Label(
                loading_frame, 
                text="Applying noise reduction...\nThis may take a moment.",
                justify='center'
            ).pack(pady=5)
            
            progress = ttk.Progressbar(loading_frame, mode='indeterminate')
            progress.pack(fill=X, pady=5)
            progress.start()
            
            self.root.update_idletasks()
            
            self.processor.reduce_noise(strength=strength)
            self.update_filter_stack_display()
            
            loading.destroy()
            
            if dialog:
                dialog.deiconify()
            
            self.display_image()
            self.update_status("Noise reduction applied")
            
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            self.handle_error(f"Error reducing noise: {str(e)}")
    
    def equalize_histogram(self):
        """Apply histogram equalization to the image"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            self.update_status("Equalizing histogram...")
            self.processor.equalize_histogram()
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Histogram equalized")
            
            if hasattr(self, 'histogram_visible') and self.histogram_visible:
                self.show_histogram(toggle=False)
                
        except Exception as e:
            self.handle_error(f"Error equalizing histogram: {str(e)}")
    
    def apply_edge_detection(self, method, dialog=None):
        """Apply edge detection with the specified method"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            if method == 'canny' and dialog is None:
                self.apply_canny_edge_detection_dialog()
                return
            
            method_name = {
                'sobel': 'Sobel',
                'laplacian': 'Laplacian',
                'canny': 'Canny'
            }.get(method, 'Edge')
            
            self.update_status(f"Applying {method_name} edge detection...")
            
            threshold1 = 100
            threshold2 = 200
            
            if dialog and method == 'canny':
                threshold1 = dialog.get('threshold1', threshold1)
                threshold2 = dialog.get('threshold2', threshold2)
            
            self.processor.apply_edge_detection(method, threshold1, threshold2)
            self.update_filter_stack_display()
            self.display_image()
            self.update_status(f"{method_name} edge detection applied")
            
        except Exception as e:
            self.handle_error(f"Error applying edge detection: {str(e)}")
    
    def apply_canny_edge_detection_dialog(self):
        """Show dialog for Canny edge detection parameters"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            dialog = ttk.Toplevel(self.root)
            dialog.title("Canny Edge Detection")
            dialog.geometry("350x200")
            dialog.transient(self.root)
            dialog.grab_set()
            
            controls_frame = ttk.Frame(dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            threshold1_frame = ttk.Frame(controls_frame)
            threshold1_frame.pack(fill=X, pady=5)
            
            threshold1_label = ttk.Label(threshold1_frame, text="Threshold 1:")
            threshold1_label.pack(side=LEFT)
            
            threshold1_var = tk.IntVar(value=100)
            threshold1_slider = ttk.Scale(
                threshold1_frame,
                from_=0,
                to=255,
                variable=threshold1_var,
                orient=HORIZONTAL
            )
            threshold1_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)

            formatted_threshold1 = tk.StringVar()

            def update_threshold1_text(*args):
                formatted_threshold1.set(f"{threshold1_var.get():.2f}")

            threshold1_var.trace_add("write", update_threshold1_text)

            threshold1_value = ttk.Label(threshold1_frame, textvariable=formatted_threshold1)
            threshold1_value.pack(side=LEFT, padx=(0, 5))
            
            threshold2_frame = ttk.Frame(controls_frame)
            threshold2_frame.pack(fill=X, pady=5)
            
            threshold2_label = ttk.Label(threshold2_frame, text="Threshold 2:")
            threshold2_label.pack(side=LEFT)
            
            threshold2_var = tk.IntVar(value=200)
            threshold2_slider = ttk.Scale(
                threshold2_frame,
                from_=0,
                to=255,
                variable=threshold2_var,
                orient=HORIZONTAL
            )
            threshold2_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)

            formatted_threshold2 = tk.StringVar()

            def update_threshold2_text(*args):
                formatted_threshold2.set(f"{threshold2_var.get():.0f}")

            threshold2_var.trace_add("write", update_threshold2_text)
            
            threshold2_value = ttk.Label(threshold2_frame, textvariable=formatted_threshold2)
            threshold2_value.pack(side=LEFT, padx=(0, 5))
            
            preview_var = tk.BooleanVar(value=False)
            preview_check = ttk.Checkbutton(
                controls_frame, 
                text="Live Preview",
                variable=preview_var
            )
            preview_check.pack(anchor=W, pady=5)
            
            original_img = self.processor.processed_image.copy()
            
            def update_preview(*args):
                if preview_var.get():
                    temp_processor = ImageProcessor()
                    temp_processor.original_image = self.processor.original_image.copy()
                    temp_processor.processed_image = original_img.copy()
                    
                    temp_processor.apply_edge_detection(
                        'canny',
                        threshold1_var.get(),
                        threshold2_var.get(),
                        preview=True
                    )
                    
                    self.processed_image_preview = temp_processor.processed_image
                    self.display_preview_image()
            
            threshold1_var.trace_add("write", update_preview)
            threshold2_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.display_image()
                dialog.destroy()
            
            cancel_btn = ttk.Button(
                button_frame,
                text="Cancel",
                command=on_cancel
            )
            cancel_btn.pack(side=RIGHT, padx=5)
            
            apply_btn = ttk.Button(
                button_frame,
                text="Apply",
                style="primary.TButton",
                command=lambda: self.apply_edge_detection(
                    'canny', 
                    {'threshold1': threshold1_var.get(), 'threshold2': threshold2_var.get()}
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.handle_error(f"Error creating Canny edge detection dialog: {str(e)}")
    
    def display_preview_image(self):
        """Display a temporary preview image"""
        if not hasattr(self, 'processed_image_preview') or self.processed_image_preview is None:
            return
        
        if self.preview_mode:
            proc_width = self.processed_canvas.winfo_width()
            proc_height = self.processed_canvas.winfo_height()
            
            pil_proc = self.processor.get_display_image(
                width=proc_width, 
                height=proc_height, 
                image=self.processed_image_preview
            )
            
            if pil_proc:
                self.tk_proc_image = ImageTk.PhotoImage(pil_proc)
                self.processed_canvas.delete("all")
                self.processed_canvas.create_image(
                    proc_width // 2,
                    proc_height // 2,
                    anchor=CENTER,
                    image=self.tk_proc_image
                )
                self.processed_canvas.create_text(
                    10, 10, 
                    text="Preview", 
                    fill="white", 
                    anchor=NW,
                    font=("Helvetica", 10, "bold")
                )
        else:
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            pil_image = self.processor.get_display_image(
                width=canvas_width, 
                height=canvas_height, 
                image=self.processed_image_preview
            )
            
            if pil_image:
                self.tk_image = ImageTk.PhotoImage(pil_image)
                self.image_canvas.delete("all")
                self.image_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor=CENTER,
                    image=self.tk_image
                )
                self.image_canvas.create_text(
                    10, 10, 
                    text="Preview", 
                    fill="white", 
                    anchor=NW,
                    font=("Helvetica", 10, "bold")
                )
    
    def apply_shear_dialog(self):
        """Show dialog for shear transformation"""
        if not self.image_loaded:
            self.handle_error("No image loaded")
            return
        
        try:
            dialog = ttk.Toplevel(self.root)
            dialog.title("Shear Transformation")
            dialog.geometry("350x200")
            dialog.transient(self.root)
            dialog.grab_set()
            
            controls_frame = ttk.Frame(dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            x_shear_frame = ttk.Frame(controls_frame)
            x_shear_frame.pack(fill=X, pady=5)
            
            x_shear_label = ttk.Label(x_shear_frame, text="X-Shear:")
            x_shear_label.pack(side=LEFT)
            
            x_shear_var = tk.DoubleVar(value=0.0)
            x_shear_slider = ttk.Scale(
                x_shear_frame,
                from_=-0.5,
                to=0.5,
                variable=x_shear_var,
                orient=HORIZONTAL
            )
            x_shear_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            formatted_x = tk.StringVar()
            
            def update_x_text(*args):
                formatted_x.set(f"{x_shear_var.get():.2f}")
            
            x_shear_var.trace_add("write", update_x_text)
            update_x_text()
            
            x_shear_value = ttk.Label(x_shear_frame, textvariable=formatted_x)
            x_shear_value.pack(side=LEFT, padx=(0, 5))
            
            y_shear_frame = ttk.Frame(controls_frame)
            y_shear_frame.pack(fill=X, pady=5)
            
            y_shear_label = ttk.Label(y_shear_frame, text="Y-Shear:")
            y_shear_label.pack(side=LEFT)
            
            y_shear_var = tk.DoubleVar(value=0.0)
            y_shear_slider = ttk.Scale(
                y_shear_frame,
                from_=-0.5,
                to=0.5,
                variable=y_shear_var,
                orient=HORIZONTAL
            )
            y_shear_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            formatted_y = tk.StringVar()
            
            def update_y_text(*args):
                formatted_y.set(f"{y_shear_var.get():.2f}")
            
            y_shear_var.trace_add("write", update_y_text)
            update_y_text()
            
            y_shear_value = ttk.Label(y_shear_frame, textvariable=formatted_y)
            y_shear_value.pack(side=LEFT, padx=(0, 5))
            
            preview_var = tk.BooleanVar(value=False)
            preview_check = ttk.Checkbutton(
                controls_frame, 
                text="Live Preview",
                variable=preview_var
            )
            preview_check.pack(anchor=W, pady=5)
            
            original_img = self.processor.processed_image.copy()
            
            def update_preview(*args):
                if preview_var.get():
                    temp_processor = ImageProcessor()
                    temp_processor.original_image = self.processor.original_image.copy()
                    temp_processor.processed_image = original_img.copy()
                    
                    temp_processor.apply_shear(
                        shear_x=x_shear_var.get(),
                        shear_y=y_shear_var.get()
                    )
                    
                    self.processed_image_preview = temp_processor.processed_image
                    self.display_preview_image()
            
            x_shear_var.trace_add("write", update_preview)
            y_shear_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.display_image()
                dialog.destroy()
            
            cancel_btn = ttk.Button(
                button_frame,
                text="Cancel",
                command=on_cancel
            )
            cancel_btn.pack(side=RIGHT, padx=5)
            
            apply_btn = ttk.Button(
                button_frame,
                text="Apply",
                style="primary.TButton",
                command=lambda: self.apply_shear(
                    x_shear_var.get(),
                    y_shear_var.get(),
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.handle_error(f"Error creating shear dialog: {str(e)}")
    
    def apply_shear(self, shear_x, shear_y, dialog=None):
        """Apply shear transformation to the image"""
        try:
            self.update_status(f"Applying shear (X:{shear_x:.2f}, Y:{shear_y:.2f})...")
            self.processor.apply_shear(shear_x=shear_x, shear_y=shear_y)
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Shear transformation applied")
            
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            self.handle_error(f"Error applying shear: {str(e)}")


def main():
    """Main function to start the application"""
    root = ttk.Window()
    app = ImageProcessingApp(root)
    
    root.bind("<Configure>", app.on_resize)
    root.bind("<Escape>", lambda e: setattr(app, 'cropping', False) 
              if hasattr(app, 'cropping') else None)
    
    root.mainloop()


if __name__ == "__main__":
    main()