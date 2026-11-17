# Demo page (user testing)

Dibuka di `http://localhost:8000/demo/`, disajikan oleh backend. Karena halaman dan videonya satu origin, canvas tidak ter-*taint* dan extension bisa membaca frame.

## Menyiapkan klip
```
demo/clips/A/clip01.mp4 ... clip05.mp4   # set A
demo/clips/B/clip01.mp4 ... clip05.mp4   # set B
```
- Ambil klip dari **aktor test** saja, yaitu wajah yang tidak pernah dilihat model saat training. Tiap set berisi 5 klip, campuran real dan fake.
- Nama file harus **netral** (`clip01` dst.), karena nama file terlihat di URL.
- Simpan kunci jawaban di `docs/user_testing/answer_key.csv` (kolom `set,clip,label,source_video`). File ini **jangan** ditaruh di folder `demo/` karena semua isi folder itu bisa dibuka lewat browser.
- File video tidak di-commit (sudah ada di `.gitignore`).
- Format **H.264 MP4** atau WebM supaya bisa diputar di Chrome.

## Alur sesi (counterbalanced)
Contoh untuk peserta ganjil: set A tanpa extension, lalu set B dengan extension. Peserta genap kebalikannya.
1. Buka `/demo/?pid=P01&set=A&phase=unassisted`. Extension **dimatikan** lewat popup.
2. Peserta menonton dan menjawab Real/Deepfake + tingkat keyakinan 1–5.
3. Ganti ke `?pid=P01&set=B&phase=assisted`, **nyalakan** extension, lalu ulangi.
4. Isi kuesioner SUS (di luar halaman ini).
5. Klik **Download all answers (CSV)** di akhir hari. Jawaban disimpan di localStorage browser itu, jadi pakai satu laptop yang sama untuk semua peserta.

Gabungkan CSV dengan `answer_key.csv` untuk menghitung akurasi peserta tanpa vs dengan extension.
