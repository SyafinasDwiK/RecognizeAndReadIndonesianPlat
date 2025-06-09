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
from PIL import Image

# --- Bagian 1: Konfigurasi karakter OCR ---
char_list = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
max_label_len = 10  # Atur sesuai dataset OCR

# --- Bagian 2: Build OCR CRNN Model ---
def build_model(char_list, max_label_len):
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

# --- Bagian 3: Fungsi preprocessing image untuk OCR ---
def pre_process_image_from_array(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    if h < 32:
        img = np.concatenate((img, np.ones((32 - h, w)) * 255))
    if w < 128:
        img = np.concatenate((img, np.ones((32, 128 - w)) * 255), axis=1)
    if w > 128 or h > 32:
        img = cv2.resize(img, (128, 32))
    img = np.expand_dims(img, axis=-1) / 255.
    return img

def predict_text_from_image(img, act_model, char_list):
    prediction = act_model.predict(np.array([img]))
    input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
    decoded = K.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]
    out = K.get_value(decoded)
    result = ""
    for p in out[0]:
        if int(p) != -1 and int(p) < len(char_list):
            result += char_list[int(p)]
    return result

# --- Bagian 4: Deteksi plat dengan YOLO ONNX ---
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad)
    im = cv2.copyMakeBorder(im, int(dh), int(dh), int(dw), int(dw), cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh

def preprocess_yolo(image, img_size=640):
    img, r, dw, dh = letterbox(image, new_shape=(img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0), r, dw, dh

def postprocess(output, shape, r, dw, dh, conf_thres=0.25):
    preds = np.squeeze(output[0]).transpose(1, 0)
    results = []
    for pred in preds:
        cx, cy, w, h, conf = pred
        if conf < conf_thres:
            continue
        x1 = (cx - w / 2 - dw) / r
        y1 = (cy - h / 2 - dh) / r
        x2 = (cx + w / 2 - dw) / r
        y2 = (cy + h / 2 - dh) / r
        results.append((int(x1), int(y1), int(x2), int(y2), conf))
    return results

# --- Bagian 5: Pipeline utama ---
def detect_and_read_plate(image_path, onnx_weights_path, ocr_weights_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Tidak dapat memuat gambar: {image_path}")
        return

    # 1. Deteksi plat dengan YOLO
    session = onnxruntime.InferenceSession(onnx_weights_path)
    input_name = session.get_inputs()[0].name
    blob, r, dw, dh = preprocess_yolo(image)
    output = session.run(None, {input_name: blob})
    detections = postprocess(output, image.shape[:2], r, dw, dh)

    if not detections:
        print("[INFO] Tidak ada plat terdeteksi.")
        return

    # 2. Crop plat terbaik (conf tertinggi)
    x1, y1, x2, y2, conf = max(detections, key=lambda x: x[4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        print("[WARNING] Crop kosong.")
        return

    # Simpan untuk debug
    cv2.imwrite("cropped_plate.jpg", crop)

    # 3. OCR CRNN
    act_model = build_model(char_list, max_label_len)
    if os.path.exists(ocr_weights_path):
        act_model.load_weights(ocr_weights_path)
        print(f"[INFO] OCR model loaded: {ocr_weights_path}")
    else:
        print("[WARNING] Tidak ada file bobot OCR ditemukan.")

    preprocessed_img = pre_process_image_from_array(crop)
    text = predict_text_from_image(preprocessed_img, act_model, char_list)
    print(f"[RESULT] Teks Plat Nomor: '{text}'")

# --- Contoh Pemakaian ---
if __name__ == "__main__":
    detect_and_read_plate(
        image_path="fotoplat\H888SI.jpg",                         # Gambar asli
        onnx_weights_path="weight/best.onnx",              # Model deteksi plat
        ocr_weights_path="weight/OCR_CRNN.keras"           # Model OCR
    )
