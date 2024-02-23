# Project Predictive Analytics By Isa Aulia Almadani - Prediksi Harga Jual Mobil dengan Machine Learning

## Domain Proyek

Dalam industri otomotif, harga jual mobil ditentukan oleh banyak faktor seperti merek, model, tahun produksi, spesifikasi mesin, transmisi, dimensi kendaraan dan lain-lain. Beberapa fitur tertentu memiliki pengaruh yang besar terhadap penetapan harga jual dibanding fitur lainnya. Pemahaman mengenai fitur-fitur utama penentu harga sangatlah penting bagi para produsen maupun dealer mobil dalam menentukan strategi penetapan harga yang optimal.

Dengan memanfaatkan data histori harga jual dan spesifikasi mobil, sistem predictive modelling dapat dibangun untuk memprediksi harga jual mobil berdasarkan fitur-fitur tertentu menggunakan machine learning. Menurut jurnal yang berjudul [Prediksi Harga Mobil Bekas dengan Machine Learning](https://jurnal.syntaxliterate.co.id/index.php/syntax-literate/article/view/2716), teknik Machine Learning dapat dimanfaatkan dalam melakukan prediksi harga mobil bekas agar menghasilkan akurasi yang cukup baik.

Beberapa algoritma machine learning yang dapat digunakan antara lain random forest, K-Nearest Neighbors atau KNN, dan Boosting. Penelitian yang berjudul [Used Car Price Prediction with Random Forest Regressor Model](https://journal.stmikjayakarta.ac.id/index.php/jisicom/article/view/752/506) memperkuat gagasan tersebut bahwa Random Forest yang digunakan dalam permodelan kasus prediksi mampu menghasilkan prediksi harga jual mobil bekas dengan akurasi yang baik. Selain itu, penelitian lain yang berjudul [Komparasi Algoritma K-NearestNeighbors dan Random Forest Pada Prediksi Harga Mobil Bekas](https://jurnal.polsri.ac.id/index.php/jupiter/article/view/5435) menunjukan bahwa penggunaan modelling K-Nearest Neighbors menghasilkan performa paling bagus dengan adanya set hyper parameter tunning neighbors. Pada penelitian [Penerapan Model Machine Learning Algoritma Gradient Boosting dan LinearRegression Melakukan Prediksi Harga Kendaraan Bekas](https://jurnal.unity-academy.sch.id/index.php/jirsi/article/view/56/44) menunjukan bahwa Algoritma boosting dengan kombinasi parameter optimal terbukti mampu meningkatkan akurasi prediksi dibandingkan model dasar.

Kesimpulan yang didapat bahwa penggunaan ketiga algoritma diatas sebagai solusi dalam membangun model prediksi. Model tersebut diharapkan dapat membantu produsen dalam menentukan harga yang kompetitif di pasar dan dealer mobil dalam merumuskan strategi penetapan harga yang optimal. Untuk proses pengembangan model kasus tersebut, dataset yang digunakan berisi fitur-fitur terkait komponen-komponen dalam mobil. Dataset tersebut disusun dengan teknik scraped dari Twitter dan Edmunds oleh penyedia sumber asal dataset yang terdapat di platform [Kaggle](https://www.kaggle.com/datasets/CooperUnion/cardataset)

## Business Understanding
### 1. Problem Statements
Berdasarkan latar belakang yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem predictive modelling untuk menjawab dua permasalahan berikut:
* Apa fitur-fitur yang paling berpengaruh terhadap harga jual mobil?
* Berapa estimasi harga jual mobil berdasarkan fitur-fitur yang paling berpengaruh dengan MRSP?
### 2. Goals
Berdasarkan problem statments yang sudah teruraikan diatas bahwa tujuan atau goalsnya sebagai berikut:
* Mengetahui fitur-fitur yang memiliki pengaruh erat terhadap harga jual.
* Membuat model machine learning yang dapat memprediksi estimasi harga jual mobil secara akurat berdasarkan fitur-fitur yang berpengaruh erat.
* Menghitung rata-rata selisih antara nilai prediksi dengan nilai y_true untuk setiap algoritma.
* Melihat akurasi model yang paling tinggi dari setiap algoritma.
### 3. Solution Statemnet
Untuk kedua problem statemnets diatas, solusi yang akan digunakan sebagai berikut:
* Membangun model prediksi untuk memperkirakan estimasi harga jual mobil berdasarkan harga_jual menggunakan algoritma machine learning, yakni **K-Nearest Neighbor(KNN)**. Model yang dibangun dengan Algoritma tersebut menggunakan teknik Grid Search untuk mencapai performa yang diinginkan
* Menggunakan algoritma machine learning **Random Forest (Bagging Algorithm)** dengan menyesuaikan beberapa parameter melalui optimasi teknik Grid Search hingga mencapai performa yang optimal.
* Menggunakan algoritma machine learning **Boosting Algorithm** dan menambahkan hyperparameter tuning dengan teknik Grid Search untuk mencapai performa model terbaik.
* Metrik yang digunakan dalam proses evaluasi model adalah [Mean Squared Eror](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) atau MSE karena dinilai cocok untuk kasus regresi.
### 4. Metodologi
Prediksi harga adalah tujuan yang ingin dicapai dan harga merupakan variabel kontinu. Dalam predictive analytics, untuk data yang bersifat variabel kontinu artinya merupakan permasalahan regresi. Oleh karena itu, metodologi pada proyek ini adalah: membangun model regresi dengan harga jual mobil sebagai variabel target(dependent variable) dan fitur-fitur utama mobil sebagai variabel prediktor(independent variabel).

## Data Understanding
Proyek ini menggunakan dataset yang bernama [Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset). Dataset ini memiliki 11.913 baris dan 16 kolom. Fitur-fitur di dalamnya terdiri dari merek, model, tahun produksi, bahan bakar, dan fitur lain. Dataset ini digunakan untuk memprediksi harga jual mobil yang berada di USA dari tahun 1990-2018. Cara pengambilan dataset ini berdasarkan sumber pengelola [Jeffrey Shih](https://www.kaggle.com/datasets/CooperUnion/cardataset) menggunakan teknik Scraped dari Emunds dan Twitter.
### Description variable
Variabel - variabel yang ada dalam dataset [Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset) sebagai berikut:
* Make: Merek/manufaktur mobil (BMW, Acura, linkoln, dll). Variabel ini berjumlah 48 nilai unik dan bertipe data kategorikal.
*	Model: Model spesifik dari mobil (1 Series M, 1 Series, dll). Variabel ini memiliki nilai unik yang banyak sekali dengan jumlah 915 unit. Variabel ini bertipe data kategorikal. 
*	Year: Tahun produksi mobil. Year/tahun produksi adalah Variabel dengan jumlah nilai unik sebanyak 28 dan variabel dengan tipe data numerik.
*	Engine Fuel Type: Jenis bahan bakar mesin (gasoline, diesel, etc). Variabel ini berjumlah 10 nilai unik dan bertipe data kategorikal.
*	Engine HP: Tenaga kuda mesin dalam satuan horsepower. Variabel ini memiliki nilai unik yang lumayan ybanyak dengan jumlah 356 unit. Variabel ini bertipe data numerik. 
*	Engine Cylinders: Jumlah silinder mesin. Engine Cyliners/jumlah silinder pada mesin adalah Variabel dengan jumlah nilai unik sebanyak 9 dan variabel dengan tipe data numerik.
*	Transmission: Tipe transmisi (manual, auto, dll). Variabel ini berjumlah 5 nilai unik dan bertipe data kategorikal.
* Type	Driven_Wheels:Roda penggerak (rear wheel drive, all wheel drive	, dll). Driven Wheels atau roda penggerak adalah Variabel dengan jumlah nilai unik sebanyak 4 dan variabel dengan tipe data numerik. 
*	Number of Doors: Jumlah pintu mobil. Variabel ini berjumlah 3 nilai unik dan bertipe data numerik.
*	Market Category: Kategori pasar mobil (Factory Tuner,Luxury,High-Performance, Luxury, dll). Variabel ini memiliki nilai unik yang cukup banyak dengan jumlah 71 unit. Variabel ini bertipe data kategorikal.
* Vehicle Size: Ukuran mobil (Compact, midsize, dll). Vehicle Size/ukuran mobil adalah Variabel dengan jumlah nilai unik sebanyak 3 dan variabel dengan tipe data kategorikal.
* Vehicle Style: Gaya/model mobil (Coupe, Convertible, dll). Variabel ini berjumlah 16 nilai unik dan bertipe data kategorikal.
* highway MPG: Jarak tempuh per galon di jalan tol. Variabel ini berjumlah 59 nilai unik dan bertipe data numerik.
* city mpg: Jarak tempuh per galon di perkotaan.  Variabel ini memiliki nilai unik yang cukup banyak dengan jumlah 69 unit. Variabel ini bertipe data numerik. 
* Popularity: Tingkat popularitas mobil.  Ini adalah Variabel dengan jumlah nilai unik sebanyak 48 dan variabel dengan tipe data numerik.
* MSRP: Harga jual yang disarankan produsen. Ini adalah variabel target(dependent variable) yang akan diprediksi dalam project ini. 

Informasi di atas diperoleh menggunakan bantuan library pandas melalui metode info(), unique(), dan nunique().
### Data Gathering dan Data Loading
Tabel 1. Dataset yang dibentuk menjadi DataFrame di Environment Colab
Dataset yang awalnya tersimpan diluar environment colab(repository github) diimport lalu dibaca dengan teknik read_csv melalui Pandas Library. DataFrame inilah yang nanti digunakan dalam prose modeling.
|index|Make|Model|Year|Engine Fuel Type|Engine HP|Engine Cylinders|Transmission Type|Driven\_Wheels|Number of Doors|Market Category|Vehicle Size|Vehicle Style|highway MPG|city mpg|Popularity|MSRP|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|BMW|1 Series M|2011|premium unleaded \(required\)|335\.0|6\.0|MANUAL|rear wheel drive|2\.0|Factory Tuner,Luxury,High-Performance|Compact|Coupe|26|19|3916|46135|
|1|BMW|1 Series|2011|premium unleaded \(required\)|300\.0|6\.0|MANUAL|rear wheel drive|2\.0|Luxury,Performance|Compact|Convertible|28|19|3916|40650|
|2|BMW|1 Series|2011|premium unleaded \(required\)|300\.0|6\.0|MANUAL|rear wheel drive|2\.0|Luxury,High-Performance|Compact|Coupe|28|20|3916|36350|
|3|BMW|1 Series|2011|premium unleaded \(required\)|230\.0|6\.0|MANUAL|rear wheel drive|2\.0|Luxury,Performance|Compact|Coupe|28|18|3916|29450|
|4|BMW|1 Series|2011|premium unleaded \(required\)|230\.0|6\.0|MANUAL|rear wheel drive|2\.0|Luxury|Compact|Convertible|28|18|3916|34500|
|5|BMW|1 Series|2012|premium unleaded \(required\)|230\.0|6\.0|MANUAL|rear wheel drive|2\.0|Luxury,Performance|Compact|Coupe|28|18|3916|31200|

## Exploratory Data Analysis or (EDA)
EDA merupakan proses fundamental dalam analytics untuk memahami dan mengeksplorasi data sebelum melakukan modelling. EDA sangat diperlukan dalam analisis karena membantu menghasilkan model yang akurat. Pada proses EDA kali ini, terdapat lima tahapan yang akan dilakukan, yaitu Check characteristic data, Data Assesing, Data Cleaning, Univariate Analysis, dan Multivariate Analysis.
### Check characteristic data
Pada tahap ini, data akan dilihat statistik ssecara umum, singkat, dan informatif. Tujuanya memberikan ringkasan statistik deskriptif dari DataFrame df_early. Ini mencakup statistik seperti mean, median, kuartil, nilai maksimum, dan nilai minimum untuk setiap kolom yang berisi data numerik. Ini adalah langkah awal yang diperlukan dalam eksplorasi dan pemahaman terhadap dataset/dataframe menggunakan metode describe().
Hasilnya adalah
### Data Assesing
Data Assesing  adalah proses evaluasi dan analisis awal terhadap data untuk memahami karakteristiknya, kualitasnya, dan potensi masalahnya sebelum melakukan analisis lebih lanjut. Pada tahap ini, terdapat tahapan-tahapan yang akan dilakukan secara rinci seperti mengecek missing or null values, mengecek duplikasi data dalam baris, dan mendeteksi outliers. 
* Checking the missing or null values
Proses ini melibatkan pengecekan apakah ada nilai yang hilang atau null dalam dataset. Nilai yang hilang dapat mengganggu analisis data karena dapat menyebabkan bias atau kesalahan dalam hasil analisis. Oleh karena itu, penting untuk mengidentifikasi di mana nilai-nilai tersebut hilang dan memutuskan bagaimana cara menangani mereka.
Dengan menggunakan bantuan metode isna() yang dijumlahkan, hasil terlihat bahwa terdapat missing value pada kolom Engine Fuel Type sebayak 3 sel, Engine HP sebanyak 69 sel, Engine Cylinders 30 sebanyak 30 sel, Number of Doors sebanyak 6 sel, dan Market Category sebanyak 3742 sel.

 
* Checking the duplicate rows
Pada tahap ini, kita memeriksa apakah ada baris data yang identik atau duplikat dalam dataset. Duplikasi data bisa menjadi masalah karena mereka dapat mempengaruhi hasil analisis statistik dengan memberikan bobot tambahan pada observasi yang sama. Mengidentifikasi dan menghapus duplikasi dapat membantu memastikan keakuratan analisis data dan mencegah distorsi dalam hasil.
* Detecting Outliers


### Data Cleaning
### Exploratory Data Analysis -Univariate Analysis
### Exploratory Data Analysis -Multivariate Analysis

## Data Preparation
### Encoding Feature Category
### Reduksi Dimensi dengan PCA
### Train-Test-Split
### Standarisasi

## Model Devlopment
### Model Devlopment Menggunakan Algoritma K-Nearest Neighbor atau KNN
### Model Devlopment Menggunakan Algoritma Random Forest
### Model Devlopment Menggunakan Algoritma Boosting Algorithm

## Evaluasi Model
### Scaling numeric features
### Count MSE for data train and test
### Plot metrix with bar chart
### Prediction target variable
### Calculates the difference between the predicted value and the y_true value
### Accuracy Model with difference algorithm

## Conclusion
