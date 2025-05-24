import ttkbootstrap as ttk
from ImageProcessingApp import ImageProcessingApp

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