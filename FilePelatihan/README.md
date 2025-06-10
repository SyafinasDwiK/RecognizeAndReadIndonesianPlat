
# 🏋️‍♂️ Tutorial Pelatihan Model


## 1️⃣ Pelatihan Deteksi Plat Nomor dengan YOLOv9
Model ini digunakan untuk mendeteksi posisi plat nomor pada gambar kendaraan.
## 🗂️ Struktur Dataset

```
dataset/
├── images/
│   ├── train/
│   └── valid/
├── labels/
│   ├── train/
│   └── valid/
└── data.yaml
```
## 🔧 Langkah Pelatihan YOLOv9

Clone repository yolov9 di drive
```
!git clone https://github.com/WongKinYiu/yolov9
%cd yolov9
```
Install library/pustaka yg di butuhkan tertera pada file requirements.txt yolov9
```
!pip install -r requirements.txt
```
Download file pre train yolov9
```
!wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt
```
Memulai pelatihan
```
!python /content/drive/MyDrive/yolov9/train.py \
--batch 16 --epochs 500 --img 640 --device 0\
--data /content/drive/MyDrive/plat-detection/data.yaml \
--weights /content/drive/MyDrive/yolov9/yolov9-t-converted.pt \
--cfg /content/drive/MyDrive/yolov9/models/detect/gelan-t.yaml \
--hyp hyp.scratch-high.yaml --patience 10
```
Convert file best.pt ke best.onnx
```
!python /content/drive/MyDrive/yolov9/export.py --weights /content/drive/MyDrive/yolov9/runs/train/exp4/weights/best.pt --img 640 --batch-size 1 --include onnx
     
```

## 2️⃣ Pelatihan Model OCR (CNN + LSTM + CTC)

Untuk membaca teks plat hasil crop dari YOLO, gunakan model OCR berbasis CNN-LSTM-CTC.

## 🗂️ Struktur Dataset

Dataset OCR berbentuk gambar plat nomor dan label teks-nya, misalnya:
```
dataset/
├── images/
│   ├── B2823FW_plate.jpg
│   ├── B2823FW_platee.jpg
│   └── ...
└── labels.txt

```
labels.txt
```
B2823FW_plate.jpg	B2823FW
B2823FW_platee.jpg	B2823FW
AB15ML_plate.jpg	AB15ML
```
## ⚙️ Langkah Pelatihan OCR (CNN + LSTM + CTC)

Buka file notebook:
> 📓 **Notebook**: [`FIX_TRAIN_CNN_LSTM_CTC.ipynb`](FIX_TRAIN_CNN_LSTM_CTC.ipynb)

Pastikan dependensi seperti TensorFlow, cv2, dan PIL terinstal:
```
pip install tensorflow opencv-python pillow
```

Buka dan jalankan semua sel dalam notebook tersebut. Proses otomatis yang akan dilakukan:

1. 📥 **Memuat dataset OCR**
2. 🧹 **Preprocessing gambar**
   - Mengubah gambar ke grayscale
   - Resize ke `128x32`
   - Normalisasi piksel [0, 1]
3. 🏗️ **Build arsitektur model CNN + LSTM + CTC**
4. 🏋️‍♀️ **Melatih model hingga konvergen**
5. 💾 **Menyimpan model ke file**:



### ✅ Output Model
Model hasil pelatihan dapat langsung digunakan untuk inferensi teks dari gambar plat dengan pipeline utama:

```python
model.load_weights("OCR_CRNN.keras")
```

