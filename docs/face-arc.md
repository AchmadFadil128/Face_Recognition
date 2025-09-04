### Face ARC — Face Recognition Attendance (POST to Web API)

Aplikasi Python untuk deteksi dan pengenalan wajah menggunakan DeepFace (model ArcFace) dan MediaPipe. Saat wajah dikenal, aplikasi akan mengirim data absensi melalui HTTP POST ke endpoint web yang sudah disiapkan.

### Fitur
- **Deteksi wajah**: MediaPipe dengan pengaturan yang dioptimalkan untuk wajah kecil.
- **Pengenalan wajah**: Embedding ArcFace via DeepFace, disimpan ke file agar tidak membuat ulang setiap kali.
- **Kirim absensi via POST**: Mengirim `studentNumber` dan `status` ke `http://localhost:3000/api/attendance/mark`.
- **Anti-duplikasi per run**: Setiap `studentNumber` hanya dikirim sekali per proses berjalan.

### Struktur Proyek (ringkas)
- `face-arc.py`: Script utama face recognition dan POST absensi.
- `database/`: Folder berisi sub-folder per siswa berisi foto referensi.
- `face_embeddings.pkl`: File cache embedding yang dibuat otomatis.
- `docs/face-arc.md`: Dokumen ini.

### Kebutuhan Sistem
- Python 3.9+ (disarankan)
- Kamera (webcam/CCTV via device index 0 atau ganti source sesuai kebutuhan)

### Dependensi Python
Install paket berikut:

```bash
pip install opencv-python mediapipe deepface scikit-learn requests
```

Catatan:
- DeepFace dapat mengunduh model saat pertama kali berjalan.
- Jika Anda menggunakan virtualenv/conda, aktifkan environment terlebih dahulu.

### Persiapan Data Wajah
Taruh foto siswa pada struktur berikut:

```
database/
  Achmad/
    foto1.jpg
    foto2.jpg
  Adel/
    foto1.jpg
  Aziz/
  Ibrahim/
```

Ketentuan singkat:
- Nama folder = nama orang yang akan dikenali (harus konsisten dengan mapping di script).
- Format gambar umum didukung: `.jpg`, `.jpeg`, `.png`, `.bmp`.
- Gunakan beberapa foto yang jelas dari tiap siswa untuk akurasi lebih baik.

### Menjalankan Aplikasi
```bash
python face-arc.py
```

Kontrol runtime:
- Tekan `q` untuk keluar.
- Tekan `r` untuk rebuild embeddings (misal setelah menambah foto baru).

### Proses Embedding
- Saat pertama kali dijalankan, script akan memindai `database/` dan membuat embedding per orang.
- Embedding disimpan ke `face_embeddings.pkl`. Jalankan lagi lebih cepat karena tidak perlu regen embedding dari awal.

### Mapping Nama ke studentNumber
Mapping manual ada di `face-arc.py` pada inisialisasi kelas:

```python
self.name_to_student_number = {
    "Achmad": "SW001",
    "Adel": "SW002",
    "Aziz": "SW003",
    "Ibrahim": "SW004",
}
```

Ubah/tambah sesuai kebutuhan. Pastikan nama kunci di mapping sama persis dengan nama folder di `database/`.

### Pengiriman Absensi (POST)
- Endpoint: `http://localhost:3000/api/attendance/mark`
- Payload JSON:

```json
{
  "studentNumber": "SW001",
  "status": "PRESENT"
}
```

Catatan perilaku:
- Aplikasi mengirim POST saat wajah terdeteksi dan dikenali.
- Untuk mencegah spam, setiap `studentNumber` hanya dikirim sekali per proses berjalan (anti-duplikasi per run).
- Tidak ada error handling lanjutan atau autentikasi sesuai spesifikasi saat ini.

### Tips Akurasi
- Tambahkan 2-5 foto per siswa dari sudut/ekspresi berbeda.
- Pastikan pencahayaan memadai dan wajah cukup besar dalam frame.
- Tekan `r` setelah menambah foto untuk regenerate embeddings.

### Masalah Umum
- Kamera tidak terbaca: pastikan device index benar (ubah `cv2.VideoCapture(0)` jika perlu).
- Modul tidak ditemukan: pastikan semua dependensi sudah terpasang di environment aktif.
- Tidak ada wajah dikenali: verifikasi struktur `database/`, kualitas foto, dan konsistensi nama folder dengan mapping.

