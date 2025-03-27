import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import time

# Tangani import DeepFace dengan error handling yang lebih baik
try:
    from deepface import DeepFace
    deepface_available = True
except Exception as e:
    print(f"Error importing DeepFace: {e}")
    deepface_available = False

# Fungsi untuk menghitung histogram manual
def calculate_histogram(image):
    hist = [0] * 256
    for row in image:
        for pixel in row:
            hist[pixel] += 1
    return hist

# Fungsi untuk menampilkan histogram
def show_histogram(hist):
    plt.figure(figsize=(10, 5))
    plt.plot(hist, color='black')
    plt.title('Histogram Wajah')
    plt.xlabel('Intensitas Piksel')
    plt.ylabel('Frekuensi')
    plt.show()

# Fungsi untuk deteksi wajah dan analisis histogram
def analyze_image(image_path):
    try:
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", f"Gagal membaca gambar: {image_path}")
            return
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah menggunakan Haar Cascade
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mendeteksi wajah: {str(e)}")
            return

        if len(faces) == 0:
            messagebox.showerror("Error", "Tidak ada wajah yang terdeteksi!")
            return

        # Ambil ROI wajah pertama
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # Normalisasi kontras (ditambahkan di sini)
        face_roi = cv2.equalizeHist(face_roi)

        # Hitung histogram manual
        hist_manual = calculate_histogram(face_roi)
        print("Histogram Manual:", hist_manual[:20], "...")  # Cetak 20 nilai pertama saja

        # Hitung histogram menggunakan OpenCV
        hist_cv = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        hist_cv = [int(val[0]) for val in hist_cv]

        # Tampilkan histogram
        show_histogram(hist_manual)

        # Tampilkan gambar wajah di GUI
        face_image = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = ImageTk.PhotoImage(face_image)
        panel.config(image=face_image)
        panel.image = face_image
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat analisis gambar: {str(e)}")

# Fungsi untuk Face ID (verifikasi wajah)
def face_id():
    if not deepface_available:
        messagebox.showerror("Error", "DeepFace tidak tersedia. Silakan periksa instalasi.")
        return

    # Cek apakah file referensi ada sebelum membuka kamera
    known_face_path = "known_face.jpg"
    if not os.path.exists(known_face_path):
        messagebox.showerror("Error", f"File {known_face_path} tidak ditemukan! Silakan upload wajah referensi terlebih dahulu.")
        return
        
    try:
        # Tampilkan pesan sebelum membuka kamera
        messagebox.showinfo("Informasi", "Kamera akan dibuka. Harap posisikan wajah Anda di depan kamera.")
        
        # Coba beberapa opsi untuk membuka kamera
        cap = None
        camera_options = [
            (0, None),  # Default
            (0, cv2.CAP_DSHOW),  # DirectShow
            (1, None),  # Alternatif index
            (1, cv2.CAP_DSHOW)   # Alternatif dengan DirectShow
        ]
        
        for idx, backend in camera_options:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, backend)
                    
                # Tunggu sebentar untuk inisialisasi kamera
                time.sleep(1)
                
                if cap.isOpened():
                    # Ambil frame uji
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"Kamera terbuka: indeks {idx}, backend {backend}")
                        break
                    else:
                        cap.release()
                else:
                    if cap is not None:
                        cap.release()
            except Exception as cam_err:
                print(f"Gagal membuka kamera {idx} dengan backend {backend}: {str(cam_err)}")
                if cap is not None:
                    cap.release()
        
        if cap is None or not cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat membuka kamera! Pastikan kamera tidak sedang digunakan oleh aplikasi lain.")
            return

        # Inisialisasi cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            messagebox.showerror("Error", "Gagal memuat classifier wajah!")
            cap.release()
            return

        # Buat jendela preview untuk menampilkan feed kamera
        cv2.namedWindow("Kamera Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Kamera Preview", 640, 480)
        
        face_detected = False
        countdown = 3  # Hitung mundur 3 detik
        start_time = cv2.getTickCount()
        
        print("Memulai loop kamera...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None or frame.size == 0:
                print("Frame tidak valid, mencoba lagi...")
                # Coba beberapa kali sebelum menyerah
                time.sleep(0.1)
                continue
            
            try:
                # Cek apakah ada wajah
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                # Reset face_detected pada setiap frame
                current_face_detected = len(faces) > 0
                
                # Gambar kotak di sekitar wajah yang terdeteksi
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Hitung waktu berjalan hanya jika wajah terdeteksi
                if current_face_detected:
                    if not face_detected:  # Jika wajah baru terdeteksi, reset timer
                        face_detected = True
                        start_time = cv2.getTickCount()
                    
                    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                    remaining = max(0, countdown - int(elapsed_time))
                    
                    status_text = f"Wajah terdeteksi! Mengambil gambar dalam {remaining} detik..."
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Jika hitung mundur selesai, ambil gambar
                    if elapsed_time > countdown:
                        break
                else:
                    face_detected = False
                    status_text = "Posisikan wajah Anda di depan kamera"
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Tampilkan frame
                cv2.imshow("Kamera Preview", frame)
                
                # Cek apakah user menekan tombol ESC untuk membatalkan
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            except Exception as frame_err:
                print(f"Error memproses frame: {str(frame_err)}")
                continue
        
        # Ambil gambar terakhir dengan wajah terdeteksi
        final_frame = frame.copy()
        cap.release()
        cv2.destroyAllWindows()
        
        if not face_detected:
            messagebox.showerror("Error", "Tidak ada wajah yang terdeteksi. Silakan coba lagi.")
            return
            
        # Buat direktori temp jika belum ada
        os.makedirs("temp", exist_ok=True)
        
        # Simpan gambar sementara
        temp_image_path = os.path.join("temp", "temp_face.jpg")
        success = cv2.imwrite(temp_image_path, final_frame)
        
        if not success or not os.path.exists(temp_image_path):
            messagebox.showerror("Error", "Gagal menyimpan gambar sementara!")
            return
        
        # Verifikasi bahwa kedua gambar dapat dibaca dengan benar
        img1 = cv2.imread(known_face_path)
        img2 = cv2.imread(temp_image_path)
        
        if img1 is None:
            messagebox.showerror("Error", "Gambar referensi tidak dapat dibaca. Silakan upload ulang wajah referensi.")
            return
            
        if img2 is None:
            messagebox.showerror("Error", "Gambar dari kamera tidak dapat dibaca.")
            return
        
        try:
            # Deteksi wajah terlebih dahulu pada gambar yang diambil
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(faces) == 0:
                messagebox.showerror("Error", "Tidak ada wajah yang terdeteksi dalam gambar yang diambil!")
                return
            
            # Verifikasi wajah menggunakan DeepFace
            try:
                # Coba metode pertama: menggunakan path file
                result = DeepFace.verify(
                    img1_path=known_face_path,
                    img2_path=temp_image_path,
                    model_name="VGG-Face", 
                    enforce_detection=False,
                    distance_metric="cosine"
                )
            except Exception as e:
                print(f"Metode pertama gagal: {str(e)}, mencoba metode alternatif...")
                # Coba metode kedua: menggunakan gambar langsung
                result = DeepFace.verify(
                    img1_path=img1,
                    img2_path=img2,
                    model_name="VGG-Face", 
                    enforce_detection=False,
                    distance_metric="cosine",
                    detector_backend="opencv"
                )
                                
            # Bandingkan wajah dengan threshold yang lebih ketat
            threshold = 0.4  # Nilai yang lebih kecil = lebih ketat
            if result["verified"] and result["distance"] < threshold:
                messagebox.showinfo("Sukses", f"Wajah terverifikasi! (Similarity: {1-result['distance']:.2f})")
                analyze_image(temp_image_path)
            else:
                messagebox.showerror("Gagal", f"Wajah tidak dikenali! (Similarity: {1-result['distance']:.2f})")
                
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat verifikasi wajah: {str(e)}")
            return
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
        return
    
# Fungsi untuk mengunggah dan menyimpan gambar referensi
def upload_reference_face():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not file_path:  # User membatalkan
            return
            
        # Baca gambar yang dipilih
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Gagal membaca gambar yang dipilih!")
            return
        
        # Deteksi wajah terlebih dahulu
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            messagebox.showerror("Error", "Tidak ada wajah yang terdeteksi dalam gambar referensi!")
            return
            
        # Ambil hanya bagian wajah untuk disimpan sebagai referensi (opsional)
        # (x, y, w, h) = faces[0]
        # face_img = image[y:y+h, x:x+w]
        # cv2.imwrite("known_face.jpg", face_img)
        
        # Atau simpan seluruh gambar
        cv2.imwrite("known_face.jpg", image)
        
        if not os.path.exists("known_face.jpg"):
            messagebox.showerror("Error", "Gagal menyimpan gambar referensi!")
            return
            
        messagebox.showinfo("Sukses", "Wajah referensi berhasil disimpan!")
        
        # Tampilkan gambar referensi di panel (opsional)
        try:
            ref_image = cv2.imread("known_face.jpg")
            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            ref_image_pil = Image.fromarray(ref_image_rgb)
            ref_image_tk = ImageTk.PhotoImage(ref_image_pil)
            panel.config(image=ref_image_tk)
            panel.image = ref_image_tk
        except Exception as show_err:
            print(f"Error menampilkan gambar referensi: {str(show_err)}")
            
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat upload referensi: {str(e)}")

# Fungsi untuk mengunggah gambar
def upload_image():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            analyze_image(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat upload gambar: {str(e)}")

# Inisialisasi GUI
def create_gui():
    global root, panel
    
    root = tk.Tk()
    root.title("Aplikasi Face Recognition")
    root.geometry("800x600")
    
    # Frame untuk tombol
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    # Tombol untuk mendaftarkan wajah referensi
    ref_face_button = tk.Button(button_frame, text="Daftarkan Wajah Referensi", command=upload_reference_face, 
                              font=("Arial", 12), bg="#4CAF50", fg="white", padx=10)
    ref_face_button.pack(side=tk.LEFT, padx=10)
    
    # Tombol untuk Face ID
    face_id_button = tk.Button(button_frame, text="Buka dengan Face ID", command=face_id,
                             font=("Arial", 12), bg="#2196F3", fg="white", padx=10)
    face_id_button.pack(side=tk.LEFT, padx=10)
    
    # Tombol untuk mengunggah gambar
    upload_button = tk.Button(button_frame, text="Unggah Gambar", command=upload_image,
                            font=("Arial", 12), bg="#FF9800", fg="white", padx=10)
    upload_button.pack(side=tk.LEFT, padx=10)
    
    # Panel untuk menampilkan gambar
    panel = tk.Label(root, bg="lightgray", width=500, height=400)
    panel.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    
    # Tampilkan pesan awal
    initial_msg = "Selamat datang! Mulai dengan mendaftarkan wajah referensi."
    panel.config(text=initial_msg, font=("Arial", 14))
    
    if not deepface_available:
        messagebox.showwarning("Peringatan", "Library DeepFace tidak tersedia. Fitur face recognition tidak akan berfungsi.")
    
    return root

if __name__ == "__main__":
    try:
        root = create_gui()
        root.mainloop()
    except Exception as e:
        print(f"Error dalam aplikasi: {str(e)}")