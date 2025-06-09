
# 🚘 Automatic License Plate Recognition (ALPR) with YOLOv9 + CRNN + CTC

Sistem ini mendeteksi dan membaca teks plat nomor kendaraan secara otomatis dari gambar menggunakan kombinasi model YOLOv9 (untuk deteksi plat) dan CNN-LSTM-CTC (untuk OCR). Proyek ini cocok untuk aplikasi seperti smart parking, pelacakan kendaraan, dan sistem keamanan.


## 📌 Fitur

🔍 Deteksi plat nomor menggunakan YOLOv9 ONNX.

🧠 OCR berbasis CNN-LSTM-CTC (CRNN) untuk membaca teks tanpa anotasi karakter.

🖼️ Proses otomatis dari gambar input → crop plat → baca teks.

✅ Akurasi tinggi dan arsitektur modular.
## Arsitektur

🧩 Arsitektur
YOLOv9 (ONNX): Deteksi lokasi plat nomor pada gambar.

CNN: Ekstraksi fitur spasial dari citra plat.

LSTM: Pemahaman urutan karakter.

CTC Loss: Mengubah prediksi menjadi string teks tanpa pelabelan posisi karakter.
## 🛠️ Instalalsi Dependensi

Python 3.8+

TensorFlow 2.x

OpenCV

NumPy

ONNX Runtime

```bash
  pip install -r requirements.txt

```
    
## 🚀 Cara Menjalankan
Simpan gambar plat ke dalam folder (misalnya fotoplat/).

Siapkan model:

best.onnx: model deteksi plat (YOLOv9).

OCR_CRNN.keras: model OCR CRNN.

Jalankan skrip:
```bash
  python Full.py

```
Output akan menampilkan:

Gambar hasil crop plat (cropped_plate.jpg)

Teks plat nomor di terminal/log
