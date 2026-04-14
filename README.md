
#  Automatic License Plate Recognition (ALPR) with YOLOv9 + CRNN + CTC

Sistem ini mendeteksi dan membaca teks plat nomor kendaraan secara otomatis dari gambar menggunakan kombinasi model YOLOv9 (untuk deteksi plat) dan CNN-LSTM-CTC (untuk OCR). Proyek ini cocok untuk aplikasi seperti smart parking, pelacakan kendaraan, dan sistem keamanan.


##  Fitur Aplikasi 

Aplikasi ini merupakan sistem Graphical User Interface (GUI) berbasis Tkinter + OpenCV yang mampu mendeteksi dan membaca teks plat nomor kendaraan dari gambar file maupun kamera secara real-time, menggunakan kombinasi model:

 Deteksi plat nomor menggunakan YOLOv9 ONNX.

 OCR berbasis CNN-LSTM-CTC (CRNN) untuk membaca teks tanpa anotasi karakter.

##  Fitur Utama

###  Input Gambar
** Opsi 1: Pilih File Gambar**
- Mendukung berbagai format gambar: `.jpg`, `.png`, `.bmp`, `.tiff`, dll.
- Gambar yang dipilih akan ditampilkan pada jendela **preview** sebelum diproses.



###  Kamera Langsung
** Opsi 2: Gunakan Kamera**
- Aplikasi dapat membuka **webcam** langsung dari antarmuka GUI.
- Tombol **"Capture & Proses"** memungkinkan pengambilan gambar real-time dari kamera untuk diproses oleh sistem OCR.



###  Deteksi & OCR Otomatis
-  **Deteksi otomatis** lokasi plat nomor kendaraan menggunakan model **YOLOv9 ONNX**.
-  Pembacaan teks plat nomor dilakukan dengan **model OCR CRNN (CNN + LSTM + CTC)**.
-  Hasil teks dari plat langsung ditampilkan di GUI secara real-time.



###  Status & Preview
-  Terdapat **progress bar animasi** selama proses berlangsung.
-  Area hasil menampilkan:
  - **Teks plat nomor yang terdeteksi**
  - **Gambar plat hasil crop**
-  Indikator status:  
  `"Sedang memproses..."`, `"Tidak terdeteksi"`, dan lain-lain.


###  Model Fleksibel
-  Mendukung pemuatan model secara dinamis:
  - `best.onnx` → model deteksi plat (YOLOv9)
  - `OCR_CRNN.keras` → model OCR teks plat (CRNN)
-  Model hanya dimuat saat dibutuhkan (**lazy-loading**) untuk menghemat memori.




## Arsitektur

🧩 Arsitektur

YOLOv9 (ONNX): Deteksi lokasi plat nomor pada gambar.

CNN: Ekstraksi fitur spasial dari citra plat.

LSTM: Pemahaman urutan karakter.

CTC Loss: Mengubah prediksi menjadi string teks tanpa pelabelan posisi karakter.
## 🛠️ Instalalsi Dependensi

```bash
  pip install -r requirements.txt

```
    
## 🚀 Cara Menjalankan Test
Simpan gambar plat ke dalam folder (misalnya fotoplat/).

Siapkan model:

best.onnx: model deteksi plat (YOLOv9).

OCR_CRNN.keras: model OCR CRNN.

Jalankan skrip:
```bash
  python test.py

```
Output akan menampilkan:

Gambar hasil crop plat (cropped_plate.jpg)

Teks plat nomor di terminal/log

## 🚀 Cara Menjalankan App

Pastikan file berikut tersedia:

app.py

weight/best.onnx

weight/OCR_CRNN.keras

Jalankan skrip:
```bash
  python app.py

```




## 📌 Contoh Output
![Image](https://github.com/user-attachments/assets/cd270f32-081d-461b-8dbd-fd2b38211d36)

