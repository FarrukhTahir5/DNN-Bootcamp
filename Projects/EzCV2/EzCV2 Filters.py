import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

class OpenCVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV Function Explorer")

        self.weights_path = ""
        self.cfg_path = ""
        self.names_path = ""

        screen_height = self.root.winfo_screenheight()
        screen_width = self.root.winfo_screenwidth()
        self.root.geometry(f"200x500+0+0")
        self.new_window = tk.Toplevel(root)
        self.new_window.geometry(f"600x{int(screen_height - 400)}+{int(screen_width/3)}+0")
        self.new_window.maxsize(800, 1000)
        self.new_window.title("Source")
        self.new_window.panel = tk.Label(self.new_window)
        self.new_window.panel.pack(padx=10, pady=10)
        self.new_window.resizable(False, False)

        btn_frame = tk.Frame(self.new_window)
        btn_frame.pack(fill=tk.X, pady=10)

        self.image_path = None
        self.video_path = None
        self.image = None
        self.video_capture = None
        self.net = None
        self.classes = None
        self.output_layers = None
        self.filter_mode = None
        self.detect_objects_flag = False
        self.running = False
        self.thread = None

        btn_select_weights = tk.Button(btn_frame, text="Select Weights", command=self.select_weights)
        btn_select_weights.pack(side=tk.LEFT, padx=10)

        btn_select_cfg = tk.Button(btn_frame, text="Select CFG", command=self.select_cfg)
        btn_select_cfg.pack(side=tk.LEFT, padx=10)

        btn_select_names = tk.Button(btn_frame, text="Select Names", command=self.select_names)
        btn_select_names.pack(side=tk.LEFT, padx=10)

        btn_select_image = tk.Button(btn_frame, text="Select Image", command=self.select_image)
        btn_select_image.pack(side=tk.LEFT, padx=10)

        btn_select_video = tk.Button(btn_frame, text="Select Video", command=self.select_video)
        btn_select_video.pack(side=tk.LEFT, padx=10)

        btn_live_video = tk.Button(btn_frame, text="Live Video", command=self.toggle_live_video)
        btn_live_video.pack(side=tk.LEFT, padx=10)

        self.new_window.protocol("WM_DELETE_WINDOW", self.close_windows)
        self.root.protocol("WM_DELETE_WINDOW", self.close_windows)
        self.create_scrollable_frame()

    def close_windows(self):
        self.running = False
        self.root.destroy()
        self.new_window.destroy()

    def select_weights(self):
        self.weights_path = filedialog.askopenfilename()
        self.load_yolo()

    def select_cfg(self):
        self.cfg_path = filedialog.askopenfilename()
        self.load_yolo()

    def select_names(self):
        self.names_path = filedialog.askopenfilename()
        self.load_yolo()
    def apply_yolo(self, image):
        if not self.net:
            messagebox.showerror("YOLO Error", "YOLO model is not loaded. Please load the model first.")
            return image

        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    if (x, y, w, h) and isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                if isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def load_yolo(self):
        if self.weights_path and self.cfg_path and self.names_path:
            try:
                self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
                self.layer_names = self.net.getLayerNames()
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                with open(self.names_path, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
                messagebox.showinfo("YOLO", "YOLO model loaded successfully.")
            except Exception as e:
                messagebox.showerror("YOLO Error", f"Error loading YOLO: {e}")

    def select_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.load_image()

    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file.")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.apply_filter(frame)
                
                if self.detect_objects_flag:
                    frame = self.apply_yolo(frame)
                
                self.display_image(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            cap.release()
            self.stop_running()

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            if self.running:
                self.stop_running()
            else:
                self.running = True
                self.thread = threading.Thread(target=self.process_video)
                self.thread.start()

    def show_live_video(self):
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not open video capture device.")
            self.stop_running()
            return

        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.apply_filter(frame)
            
            if self.detect_objects_flag:
                frame = self.apply_yolo(frame)
            
            self.display_image(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop_running()


    def stop_running(self):
        self.running = False
        self.detect_objects_flag = False
        self.filter_mode = None

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        if self.thread is not None:
            self.thread.join()
            self.thread = None

        cv2.destroyAllWindows()


    def toggle_live_video(self):
        if self.running:
            self.stop_running()
            # Re-enable other buttons
            self.update_button_state()
        else:
            # Disable other buttons
            self.update_button_state(enabled=False)
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open video capture device.")
                self.video_capture = None
                self.update_button_state()
                return
            self.running = True
            self.thread = threading.Thread(target=self.show_live_video)
            self.thread.start()

    # def select_image(self):
    #     if self.running:
    #         self.stop_running()
    #     self.image_path = filedialog.askopenfilename()
    #     if self.image_path:
    #         self.load_image()

    # def select_video(self):
    #     if self.running:
    #         self.stop_running()
    #     self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    #     if self.video_path:
    #         self.running = True
    #         self.thread = threading.Thread(target=self.process_video)
    #         self.thread.start()

    def select_image(self):
        if self.running:
            self.stop_running()
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.load_image()

    def select_video(self):
        if self.running:
            self.stop_running()
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.running = True
            self.thread = threading.Thread(target=self.process_video)
            self.thread.start()

    def update_button_state(self, enabled=True):
        for widget in self.new_window.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def load_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "No image selected.")
            return

        self.image = cv2.imread(self.image_path)
        if self.image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        if self.detect_objects_flag:
            self.image = self.apply_yolo(self.image)
        self.display_image(self.image)

    def toggle_detect_objects(self):
        self.detect_objects_flag = not self.detect_objects_flag
        if self.image_path:
            self.apply_filter_to_image()

    def apply_blur(self, image):
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        return blurred
    def apply_sobel(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return sobel
    def equalize_histogram(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return equalized
    def convert_color_space(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        return hsv, lab
    def apply_morphology(self, image):
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(image, kernel, iterations=1)
        dilation = cv2.dilate(image, kernel, iterations=1)
        return erosion, dilation
    def apply_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return thresholded
    def apply_laplacian(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        return laplacian

    def apply_filter(self, image):
        if self.filter_mode == "canny":
            image = cv2.Canny(image, 100, 200)
        elif self.filter_mode == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image = cv2.filter2D(image, -1, kernel)
        elif self.filter_mode == "gaussianblur":
            image = cv2.GaussianBlur(image, (25, 25), 0)
        elif self.filter_mode == "medianblur":
            image = cv2.medianBlur(image, 25)
        elif self.filter_mode == "bilateralfilter":
            image = cv2.bilateralFilter(image, 9, 75, 75)
        elif self.filter_mode == "sobel":
            image = self.apply_sobel(image)
        elif self.filter_mode == "laplacian":
            image = self.apply_laplacian(image)
        elif self.filter_mode == "threshold":
            image = self.apply_threshold(image)
        elif self.filter_mode == "blur":
            image = self.apply_blur(image)
        return image
    
    def display_image(self, image):
    # Fixed display size
        fixed_width = 600
        fixed_height = 400

        # Resize the image
        image = cv2.resize(image, (fixed_width, fixed_height))
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)

        # Update the display
        self.root.after(0, lambda: self.new_window.panel.config(image=image_tk))
        self.root.after(0, lambda: setattr(self.new_window.panel, 'image', image_tk))



    def apply_filter_to_image(self):
        if self.image is not None:
            image = self.image.copy()
            image = self.apply_filter(image)
            if self.detect_objects_flag:
                image = self.apply_yolo(image)
            self.display_image(image)

    def create_scrollable_frame(self):
        canvas = tk.Canvas(self.root, bg="lightgray")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        frame = ttk.Frame(canvas, style="My.TFrame")
        canvas.create_window((0, 0), window=frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        function_categories = {
        "Basic Filters": ["Canny", "Sharpen", "GaussianBlur", "Blur"],
        "Advanced Filters": ["MedianBlur", "BilateralFilter", "Sobel", "Laplacian", "Threshold"],
        "Detection": ["Toggle Object Detection"]
    }

        for category, functions in function_categories.items():
            label = tk.Label(frame, text=category, font=("Helvetica", 16, "bold"), bg="lightgray", fg="blue")
            label.pack(anchor="w", padx=10, pady=(10, 5))

            for func in functions:
                btn = tk.Button(frame, text=func, command=lambda f=func: self.set_filter_mode(f.lower()) if f.lower() != 'toggle object detection' else self.toggle_detect_objects())
                btn.pack(fill=tk.X, padx=20, pady=5)

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def set_filter_mode(self, mode):
        self.filter_mode = mode
        self.apply_filter_to_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = OpenCVApp(root)
    root.mainloop()
