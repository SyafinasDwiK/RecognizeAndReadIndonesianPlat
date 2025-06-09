import os
import cv2
import numpy as np
import onnxruntime
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Reshape, BatchNormalization, Input,
    Conv2D, MaxPool2D, Lambda, Bidirectional, LSTM
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time

class PlateOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Plat Nomor Kendaraan")
        self.root.geometry("900x750") # Menambah tinggi jendela untuk tombol baru
        self.root.configure(bg='#f0f0f0')

        # --- Path model absolut ---
        self.onnx_model_path = r"C:/Users/HP/OneDrive/SMT 8/Workshop Peng. Citra/final/weight/best.onnx"
        self.ocr_model_path = r"C:/Users/HP/OneDrive/SMT 8/Workshop Peng. Citra/final/weight/OCR_CRNN.keras"

        # Variables
        self.image_path = tk.StringVar()
        self.result_text = tk.StringVar()
        self.status_text = tk.StringVar(value="Siap untuk memproses")

        # OCR Configuration
        self.char_list = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.max_label_len = 10
        self.act_model = None
        self.onnx_session = None
        
        # --- Variabel untuk Kamera ---
        self.cap = None
        self.is_camera_on = False
        self.last_frame = None

        self.create_widgets()
        
        # --- Menangani penutupan jendela ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(main_frame, text="Aplikasi OCR Plat Nomor Kendaraan", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # --- Frame untuk Input File ---
        file_frame = ttk.LabelFrame(main_frame, text="Opsi 1: Pilih File Gambar", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(file_frame, text="Gambar:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.image_entry = ttk.Entry(file_frame, textvariable=self.image_path, width=50)
        self.image_entry.grid(row=0, column=1, padx=(5, 5), pady=2)
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_image)
        self.browse_button.grid(row=0, column=2, pady=2)
        
        self.process_file_button = ttk.Button(file_frame, text="Proses Gambar dari File", command=self.start_processing_from_file)
        self.process_file_button.grid(row=1, column=0, columnspan=3, pady=10)

        # --- Frame untuk Kontrol Kamera ---
        camera_frame = ttk.LabelFrame(main_frame, text="Opsi 2: Gunakan Kamera", padding="10")
        camera_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.camera_button = ttk.Button(camera_frame, text="Mulai Kamera", command=self.toggle_camera)
        self.camera_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.capture_button = ttk.Button(camera_frame, text="Capture & Proses", command=self.capture_and_process, state="disabled")
        self.capture_button.grid(row=0, column=1, padx=5, pady=5)

        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        image_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 10))
        self.image_label = ttk.Label(image_frame, text="Tidak ada gambar dipilih", background='white', relief='sunken', anchor='center')
        self.image_label.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        image_frame.columnconfigure(0, weight=1)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Results frame
        result_frame = ttk.LabelFrame(main_frame, text="Hasil OCR", padding="10")
        result_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(result_frame, text="Teks Plat Nomor:").grid(row=0, column=0, sticky=tk.W)
        result_entry = ttk.Entry(result_frame, textvariable=self.result_text, font=('Arial', 14, 'bold'), width=30, state='readonly')
        result_entry.grid(row=0, column=1, padx=(10, 0), pady=5)
        self.cropped_label = ttk.Label(result_frame, text="Plat terdeteksi akan muncul di sini", background='white', relief='sunken', anchor='center')
        self.cropped_label.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        result_frame.columnconfigure(1, weight=1)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        status_label = ttk.Label(status_frame, textvariable=self.status_text, foreground='blue')
        status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def toggle_camera(self):
        if self.is_camera_on:
            self.is_camera_on = False
            self.camera_button.config(text="Mulai Kamera")
            self.capture_button.config(state="disabled")
            # Mengaktifkan kembali opsi file
            self.browse_button.config(state="normal")
            self.process_file_button.config(state="normal")
            if self.cap:
                self.cap.release()
            self.image_label.config(image='', text="Kamera dimatikan")
        else:
            self.cap = cv2.VideoCapture(0) # 0 untuk webcam default
            if not self.cap.isOpened():
                messagebox.showerror("Error Kamera", "Tidak dapat membuka kamera. Pastikan kamera tidak sedang digunakan oleh aplikasi lain.")
                return
            self.is_camera_on = True
            self.camera_button.config(text="Stop Kamera")
            self.capture_button.config(state="normal")
            # Menonaktifkan opsi file
            self.browse_button.config(state="disabled")
            self.process_file_button.config(state="disabled")
            self.update_camera_feed()

    def update_camera_feed(self):
        if self.is_camera_on:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy() # Simpan frame terakhir
                # Ubah ukuran frame agar sesuai dengan label
                h, w, _ = frame.shape
                max_h, max_w = 300, 400
                scale = min(max_w/w, max_h/h)
                w_new, h_new = int(w*scale), int(h*scale)
                frame_resized = cv2.resize(frame, (w_new, h_new))
                
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                self.image_label.configure(image=img_tk, text="")
                self.image_label.image = img_tk
            self.root.after(15, self.update_camera_feed)

    def capture_and_process(self):
        if self.last_frame is not None:
            # Matikan kamera setelah capture untuk membebaskan resource
            self.toggle_camera()
            
            # Tampilkan frame yang di-capture di preview
            self.display_image(self.last_frame, is_array=True)
            self.status_text.set("Memproses frame yang di-capture...")
            
            # Mulai pemrosesan di thread terpisah
            thread = threading.Thread(target=self.process_ocr, args=(self.last_frame,))
            thread.daemon = True
            thread.start()
            self.progress.start()
        else:
            messagebox.showwarning("Capture Gagal", "Tidak ada frame untuk di-capture.")

    def browse_image(self):
        filename = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if filename:
            self.image_path.set(filename)
            self.display_image(filename)

    def display_image(self, image_source, max_size=(400, 300), is_array=False):
        try:
            if is_array:
                img_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
            else:
                image = Image.open(image_source)

            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Tidak dapat menampilkan gambar: {str(e)}")

    def start_processing_from_file(self):
        if not self.validate_inputs():
            return
        self.progress.start()
        self.status_text.set("Memproses...")
        # Jalankan pemrosesan di thread terpisah
        thread = threading.Thread(target=self.process_ocr, args=(self.image_path.get(),))
        thread.daemon = True
        thread.start()

    def validate_inputs(self):
        if not self.image_path.get():
            messagebox.showerror("Error", "Pilih gambar terlebih dahulu!")
            return False
        if not os.path.exists(self.image_path.get()):
            messagebox.showerror("Error", "File gambar tidak ditemukan!")
            return False
        return True

    def process_ocr(self, image_source):
        # image_source bisa berupa path (str) atau array gambar (np.ndarray)
        try:
            result = self.detect_and_read_plate(image_source)
            self.root.after(0, self.processing_complete, result)
        except Exception as e:
            error_msg = f"Error selama pemrosesan: {str(e)}"
            self.root.after(0, self.processing_error, error_msg)

    def processing_complete(self, result):
        self.progress.stop()
        if result:
            self.result_text.set(result)
            self.status_text.set("Pemrosesan selesai!")
            messagebox.showinfo("Sukses", f"Plat nomor terdeteksi: {result}")
        else:
            self.result_text.set("Tidak terdeteksi")
            self.status_text.set("Tidak ada plat terdeteksi")
            self.cropped_label.configure(image='', text="Plat terdeteksi akan muncul di sini")
            self.cropped_label.image = None
            messagebox.showwarning("Peringatan", "Tidak ada plat nomor yang terdeteksi!")

    def processing_error(self, error_msg):
        self.progress.stop()
        self.status_text.set("Error dalam pemrosesan")
        messagebox.showerror("Error", error_msg)
        
    def on_closing(self):
        # Pastikan kamera dilepaskan saat menutup aplikasi
        if self.is_camera_on:
            self.is_camera_on = False
            self.cap.release()
        self.root.destroy()
        
    def display_cropped_plate(self, image_array):
        try:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
            image = image.resize((200, 60), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.cropped_label.configure(image=photo, text="")
            self.cropped_label.image = photo
        except Exception as e:
            print(f"Error displaying cropped plate: {e}")

    # --- FUNGSI OCR TIDAK BERUBAH BANYAK, HANYA INPUT detect_and_read_plate ---
    
    def detect_and_read_plate(self, image_source):
        # --- PERUBAHAN: bisa menerima path atau array gambar ---
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
            if image is None:
                raise Exception(f"Tidak dapat memuat gambar dari path: {image_source}")
        else:
            image = image_source # Ini adalah array NumPy dari kamera
        
        if self.onnx_session is None:
            self.onnx_session = onnxruntime.InferenceSession(self.onnx_model_path)

        input_name = self.onnx_session.get_inputs()[0].name
        blob, r, dw, dh = self.preprocess_yolo(image)
        output = self.onnx_session.run(None, {input_name: blob})
        detections = self.postprocess(output, r, dw, dh)

        if not detections:
            return None

        x1, y1, x2, y2, conf = max(detections, key=lambda x: x[4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        self.root.after(0, self.display_cropped_plate, crop)

        if self.act_model is None:
            self.act_model = self.build_model(self.char_list, self.max_label_len)
            self.act_model.load_weights(self.ocr_model_path)

        preprocessed_img = self.pre_process_image_from_array(crop)
        text = self.predict_text_from_image(preprocessed_img, self.act_model, self.char_list)
        return text

    def build_model(self, char_list, max_label_len):
        inputs = Input(shape=(32, 128, 1), name='input_1')
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D(pool_size=(2, 1))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 1))(x)
        x = Conv2D(64, (2, 2), activation='relu')(x)
        x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        outputs = Dense(len(char_list) + 1, activation='softmax')(x)
        act_model = Model(inputs, outputs)
        return act_model

    def pre_process_image_from_array(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        if h < 32:
            add = np.ones((32 - h, w)) * 255
            img = np.concatenate((img, add))
        if w < 128:
            add = np.ones((32, 128 - w)) * 255
            img = np.concatenate((img, add), axis=1)
        if w > 128 or h > 32:
            img = cv2.resize(img, (128, 32))
        img = np.expand_dims(img, axis=-1) / 255.
        return img

    def predict_text_from_image(self, img, act_model, char_list):
        prediction = act_model.predict(np.array([img]), verbose=0)
        input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
        decoded = K.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]
        out = K.get_value(decoded)
        result = ""
        for p in out[0]:
            if int(p) != -1 and int(p) < len(char_list):
                result += char_list[int(p)]
        return result

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        im = cv2.resize(im, new_unpad)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, dw, dh

    def preprocess_yolo(self, image, img_size=640):
        img, r, dw, dh = self.letterbox(image, new_shape=(img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0), r, dw, dh

    def postprocess(self, output, r, dw, dh, conf_thres=0.25):
        preds = np.squeeze(output[0]).transpose()
        results = []
        for pred in preds:
            cx, cy, w, h, conf = pred[:5]
            if conf < conf_thres:
                continue
            x1 = (cx - w / 2 - dw) / r
            y1 = (cy - h / 2 - dh) / r
            x2 = (cx + w / 2 - dw) / r
            y2 = (cy + h / 2 - dh) / r
            results.append((int(x1), int(y1), int(x2), int(y2), conf))
        return results

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('vista') 
    except tk.TclError:
        print("Tema 'vista' tidak tersedia, menggunakan tema default.")
    style.configure('Accent.TButton', font=('Arial', 10, 'bold'), foreground='black')
    app = PlateOCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()