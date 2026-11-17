# LookTwice Chrome Extension (Manifest V3)

## Pasang (mode developer)
1. Jalankan backend (lihat `backend/README.md`) di `http://localhost:8000`.
2. Buka `chrome://extensions`, nyalakan **Developer mode**, klik **Load unpacked**, lalu pilih folder `extension/`.
3. Klik ikon LookTwice. Popup harus menampilkan "Connected · <model>".

## Cara kerja
| File | Tugas |
|---|---|
| `content.js` | Cari `<video>` yang sedang diputar dan terlihat (≥160 px). Setiap 500 ms (2 FPS), frame digambar ke canvas, diperkecil ke maks 640 px, lalu dikirim sebagai JPEG ke service worker. Badge diambil dari rata-rata 10 frame terakhir yang ada wajahnya, dan baru muncul setelah ≥3 frame. |
| `background.js` | Meneruskan frame ke `POST /predict` dan mencatat statistik (round-trip, waktu capture, latency server). Lewat service worker karena halaman https sering tidak boleh memanggil `http://localhost` langsung. |
| `overlay.css` | Badge: ✓ Likely authentic (hijau), ? Uncertain — look twice (kuning), ! Likely deepfake (merah), – No face, 🔒 Can't analyze, × Backend offline. Status selalu ditandai ikon **dan** teks, tidak hanya warna. |
| `popup.html/js` | Tombol on/off, URL backend, status koneksi, statistik latency sesi. |

Hanya satu frame per video yang diproses sekaligus. Frame berikutnya dilewati kalau request sebelumnya belum selesai, jadi antrean tidak menumpuk saat backend lambat.

## Keterbatasan (tulis di laporan)
- **Video DRM / cross-origin tanpa CORS** (Netflix, banyak embed) membuat canvas *tainted*, sehingga frame tidak bisa dibaca. Badge akan menampilkan 🔒 "Can't analyze this video". YouTube biasanya bisa dibaca, tapi perlu dicek sendiri (smoke test).
- **Fullscreen:** badge dipasang di dokumen utama, jadi tidak terlihat saat video dalam mode fullscreen.
- Hanya wajah terbesar per frame yang dianalisis.
- Backend harus berjalan di mesin yang sama (`localhost:8000`). Untuk host lain, tambahkan origin-nya di `host_permissions` di `manifest.json`.

## Yang sudah diuji
- `content.js` disuntikkan ke halaman demo (dengan shim `chrome.*`) dan dijalankan terhadap backend sungguhan dengan model kecil + detektor dummy. Hasilnya:
  - capture berjalan sekitar 2 FPS
  - badge muncul di atas video
  - video cross-origin menampilkan 🔒
  - backend mati menampilkan ×
  - keputusan authentic / uncertain / manipulated / no_face sesuai threshold
- `background.js` diuji di Node dengan mock `chrome.*` terhadap backend yang sama.
- **Belum** dipasang sebagai extension sungguhan di Chrome, dan belum dicoba di situs video nyata.
