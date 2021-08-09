# Regresi Linier (Linear Regression) menggunakan NumPy

Regresi Linier digunakan untuk menganalisis hubungan antara data (fitur) dengan 
targetnya (label)

## Contoh Kasus

Asumsi kita sedang membuat model untuk menentukan berapa seorang perlu digaji.
Berapa gaji yang perlu dibayarkan kita notasikan sebagai `X`. Sebut saja faktor yang menentukan
gaji seseorang adalah *Hard Skill*. Kita sebut faktor tersebut dengan istilah *features*. 
Berangkat dari *features* tersebut, ditulis dengan persamaan sebagai berikut:

`X = ? x Hard Skill`

Simbol `?` mengacu *weight*/parameter yang perlu kita tentukan. Hasil dari perhitungan 
antara *weight* dan *features* di atas akan menghasil prediksi gaji (X). Contoh:

- Seseorang karyawan memiliki *level Hard Skill* 8. 
  Berapa gaji yang pantas diberikan karyawan tersebut?
  
Dengan mengisi *weight* secara acak (*initial parameter*), kita bisa mengkalkulasikan 
antara *weight* dan *features* sebagai berikut:

`X = 1000 x 8`

Maka berdasarkan kalkulasi di atas hasilnya adalah 8000. Kurang lebih begitu proses 
regresi liner nantinya. Problemnya adalah *weight* tadi masih ditentukan secara acak.
Tentu akan membuat prediksinya kurang akurat. Untuk itu kita perlu menggunakan algoritme
*Gradient Descent* untuk mencari nilai *weight* yang optimal. Setelah temukan 
*weight* yang optimal, maka bisa dikalkulasikan seperti di atas.

## Instalasi
`pip install -r requirements.txt`

## Keterangan File
1. `linear_regression.ipynb`: Berisi proses langkah-langkah Regresi Linier dan Gradient Descent.
2. `linear_regression.py`: Berisi modul Regresi Linier yang mengacu ke file nomor 1
3. `compare_using_boston_data.ipynb`: Perbandingan hasil pemrosesan menggunakan module dari file nomor 2 dan module dari 
   sklearn menggunakan `boston_data`
4. `compare_using_diabetes_data.ipynb`: Perbandingan hasil pemrosesan menggunakan module dari file nomor 2 dan module dari 
   sklearn menggunakan `diabetes_data`
   
## Referensi
1. https://towardsdatascience.com/linear-regression-from-scratch-with-numpy-5485abc9f2e4
2. https://towardsdatascience.com/linear-regression-from-scratch-with-numpy-implementation-finally-8e617d8e274c