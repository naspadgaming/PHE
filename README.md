# PHE

Repository ini berisi proyek analisis ambient noise tomography untuk deteksi risiko geoteknik pada fasilitas migas.

## Tujuan

Proyek ini menampilkan simulasi:
- raw ambient noise sebelum proses,
- cross-correlation / empirical Green's function setelah proses,
- model tomografi 2D kecepatan gelombang geser (Vs) untuk mengidentifikasi zona anomali rendah.

## File utama

- [kode.ipynb](kode.ipynb) — notebook utama yang berisi analisis dan visualisasi.

## Instalasi

Pastikan Python dan package berikut tersedia:

```bash
pip install numpy matplotlib obspy
```

## Menjalankan notebook

Buka [kode.ipynb](kode.ipynb) di Jupyter atau VS Code dan jalankan semua sel.

## Arti variabel utama

Dalam notebook, baris berikut digunakan untuk menghasilkan model Vs yang sudah dikoreksi:

```python
Vs_final = Vs - anomaly
```

Artinya nilai `Vs` dikurangi oleh `anomaly`, sehingga zona dengan kecepatan rendah dapat terlihat lebih jelas sebagai potensi area yang berisiko.

## Catatan

Proyek ini merupakan simulasi visual dan edukasi untuk menampilkan pola tomografi geoteknik dan interpretasi risiko.
