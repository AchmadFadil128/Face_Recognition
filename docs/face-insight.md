# 📌 Face Recognition System

Face Recognition System adalah program berbasis **DeepFace + OpenCV** yang digunakan untuk mendeteksi dan mengenali wajah secara real-time menggunakan webcam, CCTV, atau video file.  
Program ini dirancang untuk mendukung proyek **SIPINTAR (Sistem Pemantauan Interaktif dan Pintar)** dalam melakukan absensi otomatis dan identifikasi siswa.  

## Fitur
- Deteksi wajah dengan bundle**Insight Face** model buffalo_l.  
- Pembuatan dan penyimpanan **face embeddings** ke file `.pkl` agar lebih cepat.  
- Mendukung input dari **kamera/webcam**, **video file**, maupun **folder database foto**.  
- Threshold berbasis **cosine similarity** untuk menentukan kecocokan wajah.  
- Opsi untuk **skip frame** agar lebih ringan di CPU-only environment.  
- Konfigurasi dapat diatur langsung melalui **CLI arguments**.  

## Persyaratan
Install dependensi berikut:

```bash
pip install deepface opencv-python numpy scikit-learn
````

Jika ingin CPU-only PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Cara Menjalankan

### 1. Menjalankan dengan Webcam

```bash
python face_recognition.py --camera 0
```

### 2. Menggunakan File Video

```bash
python face_recognition.py --camera video.mp4
```

### 3. Membuat Embeddings Baru

```bash
python face_recognition.py --index --ref database
```

### 4. Menentukan Threshold & Frame Skip

```bash
python face_recognition.py --camera 0 --threshold 0.5 --frame_skip 3
```

## Argumen Lengkap

* `--camera` → Sumber input: `0` (webcam), index kamera lain, atau path video.
* `--ref` → Folder berisi gambar wajah referensi (default: `database/`).
* `--embeddings` → Lokasi file `.pkl` embeddings (default: `face_embeddings.pkl`).
* `--index` → Buat ulang embeddings dari folder referensi.
* `--frame_skip` → Lewati N frame sebelum diproses (default: 2, untuk optimasi kecepatan).
* `--threshold` → Batas kecocokan wajah (0.0–1.0, default: 0.45).
* `--min_face` → Abaikan wajah kecil di bawah ukuran tertentu (default: 30 px).

## Contoh Penggunaan

* **Absensi otomatis siswa** menggunakan kamera CCTV atau webcam.
* **Identifikasi individu** di ruang terbatas (kelas, kantor, lab).
* **Pengembangan prototype AI** untuk sistem keamanan atau monitoring.

## Pengembang

Program ini dikembangkan sebagai bagian dari proyek **SIPINTAR** dalam lomba **AI Hackathon Jawa Barat 2025**.

---