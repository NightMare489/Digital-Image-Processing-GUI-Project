import tkinter as tk
from tkinter import messagebox, simpledialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from ImageProcessor import ImageProcessor

class ImageDialogs:
    """Class for handling all dialog windows for image processing operations"""
    
    def __init__(self, app):
        """
        Initialize the dialogs class
        
        Args:
            app: The parent ImageProcessingApp instance
        """
        self.app = app
        self.root = app.root
        self.processor = app.processor
    
    def resize_image_dialog(self):
        """Show dialog to resize image"""
        if not self.app.image_loaded:
            self.app.handle_error("No image loaded")
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
                command=lambda: self.app.apply_resize(width_var.get(), height_var.get(), resize_dialog)
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.app.handle_error(f"Error creating resize dialog: {str(e)}")
    
    def adjust_brightness_contrast_dialog(self):
        """Show dialog to adjust brightness and contrast"""
        if not self.app.image_loaded:
            self.app.handle_error("No image loaded")
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
                    self.app.display_image()
            
            brightness_var.trace_add("write", update_preview)
            contrast_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.app.display_image()
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
                command=lambda: self.app.apply_brightness_contrast(
                    brightness_var.get(), 
                    contrast_var.get(), 
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.app.handle_error(f"Error creating brightness/contrast dialog: {str(e)}")
    
    def adjust_color_balance_dialog(self):
        """Show dialog to adjust color balance"""
        if not self.app.image_loaded:
            self.app.handle_error("No image loaded")
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
                    self.app.display_image()
            
            r_factor_var.trace_add("write", update_preview)
            g_factor_var.trace_add("write", update_preview)
            b_factor_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.app.display_image()
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
                command=lambda: self.app.apply_color_balance(
                    r_factor_var.get(),
                    g_factor_var.get(),
                    b_factor_var.get(),
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.app.handle_error(f"Error creating color balance dialog: {str(e)}")
    
    def adjust_hue_saturation_dialog(self):
        """Show dialog to adjust hue and saturation"""
        if not self.app.image_loaded:
            self.app.handle_error("No image loaded")
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
                    self.app.display_image()
            
            hue_var.trace_add("write", update_preview)
            sat_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.app.display_image()
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
                command=lambda: self.app.apply_hue_saturation(
                    hue_var.get(), 
                    sat_var.get(), 
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.app.handle_error(f"Error creating hue/saturation dialog: {str(e)}")
    
    def apply_gaussian_blur_dialog(self):
        """Show dialog to apply Gaussian blur"""
        if not self.app.image_loaded:
            self.app.handle_error("No image loaded")
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
                    self.app.display_image()
            
            kernel_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.processor.processed_image = original_img.copy()
                    self.app.display_image()
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
                command=lambda: self.app.apply_gaussian_blur(
                    kernel_var.get() if kernel_var.get() % 2 == 1 else kernel_var.get() + 1, 
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.app.handle_error(f"Error creating Gaussian blur dialog: {str(e)}")
    
    def apply_canny_edge_detection_dialog(self):
        """Show dialog for Canny edge detection parameters"""
        if not self.app.image_loaded:
            self.app.handle_error("No image loaded")
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
                    
                    self.app.processed_image_preview = temp_processor.processed_image
                    self.app.display_preview_image()
            
            threshold1_var.trace_add("write", update_preview)
            threshold2_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.app.display_image()
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
                command=lambda: self.app.apply_edge_detection(
                    'canny', 
                    {'threshold1': threshold1_var.get(), 'threshold2': threshold2_var.get()}
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.app.handle_error(f"Error creating Canny edge detection dialog: {str(e)}")
    
    def apply_shear_dialog(self):
        """Show dialog for shear transformation"""
        if not self.app.image_loaded:
            self.app.handle_error("No image loaded")
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
                    
                    self.app.processed_image_preview = temp_processor.processed_image
                    self.app.display_preview_image()
            
            x_shear_var.trace_add("write", update_preview)
            y_shear_var.trace_add("write", update_preview)
            preview_var.trace_add("write", update_preview)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack(fill=X, pady=10)
            
            def on_cancel():
                if preview_var.get():
                    self.app.display_image()
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
                command=lambda: self.app.apply_shear(
                    x_shear_var.get(),
                    y_shear_var.get(),
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.app.handle_error(f"Error creating shear dialog: {str(e)}")
    
    def apply_object_detection_dialog(self):
        """Show dialog for object detection parameters"""
        if not self.app.image_loaded:
            self.app.handle_error("No image loaded")
            return
        
        try:
            dialog = ttk.Toplevel(self.root)
            dialog.title("Object Detection")
            dialog.geometry("350x250")
            dialog.transient(self.root)
            dialog.grab_set()
            
            controls_frame = ttk.Frame(dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            conf_frame = ttk.Frame(controls_frame)
            conf_frame.pack(fill=X, pady=5)
            
            conf_label = ttk.Label(conf_frame, text="Confidence Threshold:")
            conf_label.pack(side=LEFT)
            
            conf_var = tk.DoubleVar(value=0.25)
            conf_slider = ttk.Scale(
                conf_frame,
                from_=0.0,
                to=1.0,
                variable=conf_var,
                orient=HORIZONTAL,
                command=lambda val: conf_var.set(float(val).__round__(2))
            )
            conf_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            formatted_conf = tk.StringVar()
            
            def update_conf_text(*args):
                formatted_conf.set(f"{conf_var.get():.2f}")
            
            conf_var.trace_add("write", update_conf_text)
            update_conf_text()
            
            conf_value = ttk.Label(conf_frame, textvariable=formatted_conf)
            conf_value.pack(side=LEFT, padx=(0, 5))
            
            labels_frame = ttk.Frame(controls_frame)
            labels_frame.pack(fill=X, pady=5)
            
            hide_labels_var = tk.BooleanVar(value=False)
            hide_labels_check = ttk.Checkbutton(
                labels_frame, 
                text="Hide Labels",
                variable=hide_labels_var
            )
            hide_labels_check.pack(side=LEFT, padx=5)
            
            hide_conf_var = tk.BooleanVar(value=False)
            hide_conf_check = ttk.Checkbutton(
                labels_frame, 
                text="Hide Confidence",
                variable=hide_conf_var
            )
            hide_conf_check.pack(side=LEFT, padx=5)
            
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
                command=lambda: self.app.apply_object_detection(
                    conf_var.get(),
                    None,
                    hide_labels_var.get(),
                    hide_conf_var.get(),
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.app.handle_error(f"Error creating object detection dialog: {str(e)}")
    
    def apply_instance_segmentation_dialog(self):
        """Show dialog for instance segmentation parameters"""
        if not self.app.image_loaded:
            self.app.handle_error("No image loaded")
            return
        
        try:
            dialog = ttk.Toplevel(self.root)
            dialog.title("Instance Segmentation")
            dialog.geometry("400x350")
            dialog.transient(self.root)
            dialog.grab_set()
            
            controls_frame = ttk.Frame(dialog, padding=10)
            controls_frame.pack(fill=BOTH, expand=YES)
            
            # Class selection
            class_frame = ttk.Frame(controls_frame)
            class_frame.pack(fill=X, pady=5)
            
            class_label = ttk.Label(class_frame, text="Class:")
            class_label.pack(side=LEFT)
            
            # Load YOLO model to get class names
            self.processor._load_yolo_model()
            class_names = {idx: name for idx, name in self.processor.yolo_model.names.items()}
            
            # Add "All Classes" option
            classes_with_all = {-1: "All Classes"}
            classes_with_all.update(class_names)
            
            class_var = tk.IntVar(value=-1)  # Default to All Classes
            class_dropdown = ttk.Combobox(
                class_frame,
                textvariable=class_var,
                state="readonly"
            )
            
            # Format class options as "ID: Name"
            class_options = [f"{idx}: {name}" for idx, name in classes_with_all.items()]
            class_dropdown['values'] = class_options
            class_dropdown.current(0)  # Set to "All Classes"
            class_dropdown.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            # Extract class ID from selection
            def get_selected_class_id():
                selection = class_dropdown.get()
                if selection.startswith("-1:"):
                    return None
                return int(selection.split(":")[0])
            
            # Visualization mode selection
            mode_frame = ttk.Frame(controls_frame)
            mode_frame.pack(fill=X, pady=5)
            
            mode_label = ttk.Label(mode_frame, text="Visualization Mode:")
            mode_label.pack(side=LEFT)
            
            mode_var = tk.StringVar(value="highlight")
            
            highlight_radio = ttk.Radiobutton(
                mode_frame,
                text="Highlight",
                variable=mode_var,
                value="highlight"
            )
            highlight_radio.pack(anchor=W, padx=5, pady=2)
            
            blur_radio = ttk.Radiobutton(
                mode_frame,
                text="Blur",
                variable=mode_var,
                value="blur"
            )
            blur_radio.pack(anchor=W, padx=5, pady=2)
            
            isolate_radio = ttk.Radiobutton(
                mode_frame,
                text="Isolate",
                variable=mode_var,
                value="isolate"
            )
            isolate_radio.pack(anchor=W, padx=5, pady=2)
            
            
            # Blur strength (only visible when blur mode is selected)
            strength_frame = ttk.Frame(controls_frame)
            strength_frame.pack(fill=X, pady=5)
            
            strength_label = ttk.Label(strength_frame, text="Blur Strength:")
            strength_label.pack(side=LEFT)
            
            strength_var = tk.IntVar(value=15)
            strength_slider = ttk.Scale(
                strength_frame,
                from_=3,
                to=51,
                variable=strength_var,
                command=lambda val: strength_var.set(
                    int(float(val)) if int(float(val)) % 2 == 1 else int(float(val)) + 1
                ),
                orient=HORIZONTAL
            )
            strength_slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
            
            strength_value = ttk.Label(strength_frame, textvariable=strength_var)
            strength_value.pack(side=LEFT, padx=(0, 5))
            
            # Show/hide strength based on mode
            def update_strength_visibility(*args):
                if mode_var.get() == "blur":
                    strength_frame.pack(fill=X, pady=5)
                else:
                    strength_frame.pack_forget()
            
            mode_var.trace_add("write", update_strength_visibility)
            update_strength_visibility()  # Initial update
            

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
                command=lambda: self.app.apply_instance_segmentation(
                    get_selected_class_id(),
                    mode_var.get(),
                    strength_var.get(),
                    False,
                    dialog
                )
            )
            apply_btn.pack(side=RIGHT, padx=5)
            
        except Exception as e:
            self.app.handle_error(f"Error creating instance segmentation dialog: {str(e)}")