import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Dict
from ultralytics import YOLO
from FilterStack import Filter, FilterStack

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
        
        # Initialize YOLO model (will be loaded on first use)
        self.yolo_model = None
        self.detection_results = None
        
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
            'histogram_equalization': self._apply_histogram_equalization,
            'edge_detection': self._apply_edge_detection,
            'shear': self._apply_shear,
            'object_detection': self._apply_object_detection,
            'instance_segmentation': self._apply_instance_segmentation
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
    
    def _load_yolo_model(self, model_name='yolov8n.pt'):
        """Load YOLO model if not already loaded"""
        if self.yolo_model is None:
            try:
                self.yolo_model = YOLO(model_name)
                return True
            except Exception as e:
                raise ValueError(f"Failed to load YOLO model: {str(e)}")
        return True
    
    def _load_yolo_segmentation_model(self, model_name='yolov8m-seg.pt'):
        """Load YOLO segmentation model if not already loaded"""
        if not hasattr(self, 'yolo_seg_model') or self.yolo_seg_model is None:
            try:
                self.yolo_seg_model = YOLO(model_name)
                return True
            except Exception as e:
                raise ValueError(f"Failed to load YOLO segmentation model: {str(e)}")
        return True
    
    def _apply_object_detection(self, params: Dict[str, Any]) -> None:
        """Apply object detection using YOLO"""
        if self.processed_image is None:
            return
        
        conf_threshold = params.get('conf_threshold', 0.25)
        classes = params.get('classes', None)  # List of class ids to detect
        hide_labels = params.get('hide_labels', False)
        hide_conf = params.get('hide_conf', False)
        
        # Load YOLO model if not already loaded
        self._load_yolo_model(params.get('model_name', 'yolov8n.pt'))
        
        # Convert image to format expected by YOLO
        image_for_detection = self.processed_image.copy()
        
        # Run detection
        self.detection_results = self.yolo_model(image_for_detection, conf=conf_threshold, classes=classes)
        
        # Draw detection results on image
        annotated_image = self.detection_results[0].plot(
            line_width=1,
            labels=not hide_labels,
            conf=not hide_conf
        )
        
        # Update processed image with detections
        self.processed_image = annotated_image
    
    
    def _apply_instance_segmentation(self, params: Dict[str, Any]) -> None:
        """Apply instance segmentation using YOLO segmentation model"""
        if self.processed_image is None:
            return
        
        # Load YOLO segmentation model if not already loaded
        self._load_yolo_segmentation_model(params.get('model_name', 'yolov8m-seg.pt'))
        
        # Convert image to format expected by YOLO
        image_for_segmentation = self.processed_image.copy()
        
        # Run segmentation
        results = self.yolo_seg_model(image_for_segmentation)

        if results[0] is None or results[0].masks is None:
            return
        
        # Extract masks and classes
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # Create mask based on selected class
        class_id = params.get('class_id', None)
        mask_mode = params.get('mask_mode', 'highlight')
        mask_strength = params.get('mask_strength', 15)
        hide_labels = params.get('hide_labels', False)
        
        combined_mask = np.zeros(self.processed_image.shape[:2], dtype=np.uint8)
        original_height, original_width = self.processed_image.shape[:2]

        for i, mask in enumerate(masks):
            if class_id is None or classes[i] == class_id:
                # Resize the mask to match the original image size
                resized_mask = cv2.resize(mask.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, resized_mask * 255)

        
        # Apply mask according to selected mode
        if mask_mode == 'highlight':
            # Create a colored highlight effect
            colored_mask = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            colored_mask[:, :, 0] = 0  # Set blue channel to 0
            colored_mask[:, :, 1] = combined_mask  # Green channel
            colored_mask[:, :, 2] = 0  # Set red channel to 0
            
            # Blend with original image
            alpha = 0.3
            self.processed_image = cv2.addWeighted(
                self.processed_image, 1, colored_mask, alpha, 0
            )
            
        elif mask_mode == 'blur':
            # Blur only the detected objects
            blurred = cv2.GaussianBlur(
                self.processed_image, 
                (mask_strength, mask_strength), 
                0
            )
            mask_inv = cv2.bitwise_not(combined_mask)
            
            # Keep original where mask is 0, use blurred where mask is 255
            bg = cv2.bitwise_and(self.processed_image, self.processed_image, mask=mask_inv)
            fg = cv2.bitwise_and(blurred, blurred, mask=combined_mask)
            self.processed_image = cv2.add(bg, fg)
            
        elif mask_mode == 'isolate':
            # Keep only the detected objects, make the rest gray
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            mask_inv = cv2.bitwise_not(combined_mask)
            
            # Use original where mask is 255, use gray where mask is 0
            fg = cv2.bitwise_and(self.processed_image, self.processed_image, mask=combined_mask)
            bg = cv2.bitwise_and(gray, gray, mask=mask_inv)
            self.processed_image = cv2.add(bg, fg)
            
    
    def apply_object_detection(self, conf_threshold=0.25, classes=None, hide_labels=False, hide_conf=False, model_name='yolov8n.pt'):
        """Detect objects in the image using YOLO and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {
            'conf_threshold': conf_threshold,
            'classes': classes,
            'hide_labels': hide_labels,
            'hide_conf': hide_conf,
            'model_name': model_name
        }
        
        class_str = ""
        if classes:
            class_names = self._get_class_names(classes)
            class_str = f" ({', '.join(class_names)})"
        
        filter_name = f"Object Detection{class_str}"
        return self.add_filter(filter_name, 'object_detection', params)
    
    
    def apply_instance_segmentation(self, class_id=None, mask_mode='highlight', mask_strength=15, hide_labels=False):
        """Apply instance segmentation and add to filter stack"""
        if self.processed_image is None:
            raise ValueError("No image loaded")
        
        params = {
            'class_id': class_id,
            'mask_mode': mask_mode,
            'mask_strength': mask_strength,
            'hide_labels': hide_labels
        }
        
        class_str = ""
        if class_id is not None:
            class_names = self._get_class_names([class_id])
            class_str = f" ({class_names[0]})"
        
        mode_str = mask_mode.capitalize().replace('_', ' ')
        filter_name = f"Instance Segmentation{class_str} - {mode_str}"
        return self.add_filter(filter_name, 'instance_segmentation', params)
    
    def _get_class_names(self, class_ids):
        """Get class names from class IDs"""
        if self.yolo_model is None:
            self._load_yolo_model()
        
        names = self.yolo_model.names
        return [names.get(cid, f"Class {cid}") for cid in class_ids]
    
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