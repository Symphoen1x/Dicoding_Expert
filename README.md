# Project Predictive Analytics By Isa Aulia Almadani - Prediksi Harga Jual Mobil dengan Machine Learning

## Domain Proyek

Dalam industri otomotif, harga jual mobil ditentukan oleh banyak faktor seperti merek, model, tahun produksi, spesifikasi mesin, transmisi, dimensi kendaraan dan lain-lain. Beberapa fitur tertentu memiliki pengaruh yang besar terhadap penetapan harga jual dibanding fitur lainnya. Pemahaman mengenai fitur-fitur utama penentu harga sangatlah penting bagi para produsen maupun dealer mobil dalam menentukan strategi penetapan harga yang optimal.

Dengan memanfaatkan data histori harga jual dan spesifikasi mobil, sistem predictive modelling dapat dibangun untuk memprediksi harga jual mobil berdasarkan fitur-fitur tertentu menggunakan machine learning. Menurut jurnal yang berjudul [Prediksi Harga Mobil Bekas dengan Machine Learning](https://jurnal.syntaxliterate.co.id/index.php/syntax-literate/article/view/2716), teknik Machine Learning dapat dimanfaatkan dalam melakukan prediksi harga mobil bekas agar menghasilkan akurasi yang cukup baik.

Beberapa algoritma machine learning yang dapat digunakan antara lain random forest, K-Nearest Neighbors atau KNN, dan Boosting. Penelitian [Used Car Price Prediction with Random Forest Regressor Model](https://journal.stmikjayakarta.ac.id/index.php/jisicom/article/view/752/506) memperkuat gagasan tersebut bahwa Random Forest yang digunakan dalam permodelan kasus prediksi mampu menghasilkan prediksi harga jual mobil bekas dengan akurasi yang baik. Selain itu, penelitian lain [Komparasi Algoritma K-NearestNeighbors dan Random Forest Pada Prediksi Harga Mobil Bekas](https://jurnal.polsri.ac.id/index.php/jupiter/article/view/5435) menunjukan bahwa penggunaan modelling K-Nearest Neighbors menghasilkan performa paling bagus dengan adanya set hyper parameter tunning neighbors. Pada penelitian [Penerapan Model Machine Learning Algoritma Gradient Boosting dan LinearRegression Melakukan Prediksi Harga Kendaraan Bekas](https://jurnal.unity-academy.sch.id/index.php/jirsi/article/view/56/44) menunjukan bahwa Algoritma boosting dengan kombinasi parameter optimal terbukti mampu meningkatkan akurasi prediksi dibandingkan model dasar.

Kesimpulan yang didapat bahwa penggunaan ketiga algoritma diatas sebagai solusi dalam membangun model prediksi. Model tersebut diharapkan dapat membantu produsen dalam menentukan harga yang kompetitif di pasar dan dealer mobil dalam merumuskan strategi penetapan harga yang optimal. Untuk proses pengembangan model kasus tersebut, dataset yang digunakan berisi fitur-fitur terkait komponen-komponen dalam mobil. Dataset tersebut disusun oleh penyedia dengan teknik scraped dari Twitter dan Edmunds. Berikut sumber asal dataset yang terdapat di platform [Kaggle](https://www.kaggle.com/datasets/CooperUnion/cardataset)

## Business Understanding
### 1. Problem Statements
Berdasarkan latar belakang yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem predictive modelling untuk menjawab dua permasalahan berikut:
1. Apa fitur-fitur yang paling berpengaruh terhadap harga jual mobil?
2. Berapa estimasi harga jual mobil berdasarkan fitur-fitur yang paling berpengaruh dengan MRSP?
### 2. Goals
Berdasarkan problem statments yang sudah teruraikan diatas bahwa tujuan atau goalsnya sebagai berikut:
1. Mengetahui fitur-fitur yang memiliki pengaruh erat terhadap harga jual.
2. Membuat model machine learning yang dapat memprediksi estimasi harga jual mobil secara akurat berdasarkan fitur-fitur yang berpengaruh erat.
3. Menghitung rata-rata selisih antara nilai prediksi dengan nilai y_true untuk setiap algoritma.
4. Melihat akurasi model yang paling tinggi dari setiap algoritma.
### 3. Solution Statemnet
Untuk kedua problem statemnets diatas, solusi yang akan digunakan sebagai berikut:
1. Membangun model prediksi untuk memperkirakan estimasi harga jual mobil berdasarkan harga_jual menggunakan algoritma machine learning, yakni **K-Nearest Neighbor(KNN)**. Model yang dibangun dengan Algoritma ini disesuaikan parameternya untuk mencapai performa yang diinginkan
2. Menggunakan algoritma machine learning **Random Forest (Bagging Algorithm)** dengan menyesuaikan parameter hingga mencapai performa yang optimal.
3. Menggunakan algoritma machine learning **Boosting Algorithm**. Model yang dibangun dengan algoritma ini disesuaikan masing-masing parameternya hingga mendapat performa yang bagus.
