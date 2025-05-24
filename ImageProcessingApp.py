import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
from PIL import ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ImageProcessor import ImageProcessor
from ImageDialogs import ImageDialogs

class ImageProcessingApp:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Modern Image Processing")
        self.root.geometry("1200x800")
        
        self.style = ttk.Style("darkly")
        self.processor = ImageProcessor()
        
        # self._create_menu()
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
        
        # Initialize dialogs
        self.dialogs = ImageDialogs(self)
       
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
            command=self.save_image_as,
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

        shear_btn = ttk.Button(
            transform_frame,
            text="Shear Transform",
            command=self.apply_shear_dialog
        )
        shear_btn.pack(pady=5, fill=X, padx=5)
        
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
                
        object_detection_btn = ttk.Button(
            advanced_frame,
            text="Object Detection",
            command=self.apply_object_detection_dialog
        )
        object_detection_btn.pack(pady=5, fill=X, padx=5)
         
        instance_segmentation_btn = ttk.Button(
            advanced_frame,
            text="Instance Segmentation",
            command=self.apply_instance_segmentation_dialog
        )
        instance_segmentation_btn.pack(pady=5, fill=X, padx=5)
        
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
    
    def apply_object_detection(self, conf_threshold, classes, hide_labels, hide_conf, dialog=None):
        """Apply object detection to the image"""
        try:
            self.update_status(f"Applying object detection (conf: {conf_threshold:.2f})...")
            self.processor.apply_object_detection(
                conf_threshold=conf_threshold,
                classes=classes,
                hide_labels=hide_labels,
                hide_conf=hide_conf
            )
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Object detection applied")
            
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            self.handle_error(f"Error applying object detection: {str(e)}")
      
    def apply_instance_segmentation(self, class_id, mask_mode, mask_strength, hide_labels=False, dialog=None):
        """Apply instance segmentation to the image"""
        try:
            mode_str = mask_mode.capitalize().replace('_', ' ')
            self.update_status(f"Applying instance segmentation ({mode_str})...")
            
            self.processor.apply_instance_segmentation(
                class_id=class_id,
                mask_mode=mask_mode,
                mask_strength=mask_strength,
                hide_labels=hide_labels
            )
            
            self.update_filter_stack_display()
            self.display_image()
            self.update_status("Instance segmentation applied")
            
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            self.handle_error(f"Error applying instance segmentation: {str(e)}")

    
    def resize_image_dialog(self):
        """Show dialog to resize image"""
        self.dialogs.resize_image_dialog()

    def apply_instance_segmentation_dialog(self):
        """Show dialog for instance segmentation parameters"""
        self.dialogs.apply_instance_segmentation_dialog()

    def apply_object_detection_dialog(self):
        """Show dialog for object detection parameters"""
        self.dialogs.apply_object_detection_dialog()

    def apply_shear_dialog(self):
        """Show dialog for shear transformation"""
        self.dialogs.apply_shear_dialog()

    def apply_canny_edge_detection_dialog(self):
        """Show dialog for Canny edge detection parameters"""
        self.dialogs.apply_canny_edge_detection_dialog()

    def apply_gaussian_blur_dialog(self):
        """Show dialog to apply Gaussian blur"""
        self.dialogs.apply_gaussian_blur_dialog()

    def adjust_color_balance_dialog(self):
        """Show dialog to adjust color balance"""
        self.dialogs.adjust_color_balance_dialog()
    
    def adjust_hue_saturation_dialog(self):
        """Show dialog to adjust hue and saturation"""
        self.dialogs.adjust_hue_saturation_dialog()

    def adjust_brightness_contrast_dialog(self):
        """Show dialog to adjust brightness and contrast"""
        self.dialogs.adjust_brightness_contrast_dialog()