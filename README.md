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
Pada tahap ini, data akan dilihat statistik ssecara umum, singkat, dan informatif. Tujuanya memberikan ringkasan statistik deskriptif dari DataFrame df_early. Ini mencakup statistik seperti mean, median, kuartil atas 75%, kuartil tengah 50%, kuartil bawah 25%, standar devisiasi,  nilai maksimum, dan nilai minimum untuk setiap kolom yang berisi data numerik. Ini adalah langkah awal yang diperlukan dalam eksplorasi dan pemahaman terhadap dataset/dataframe menggunakan metode describe().
Count adalah jumlah sampel pada data.
### Data Assesing
Data Assesing  adalah proses evaluasi dan analisis awal terhadap data untuk memahami karakteristiknya, kualitasnya, dan potensi masalahnya sebelum melakukan analisis lebih lanjut. Pada tahap ini, terdapat tahapan-tahapan yang akan dilakukan secara rinci seperti mengecek missing or null values, mengecek duplikasi data dalam baris, dan mendeteksi outliers. 
* Checking the missing or null values
Proses ini melibatkan pengecekan apakah ada nilai yang hilang atau null dalam dataset. Nilai yang hilang dapat mengganggu analisis data karena dapat menyebabkan bias atau kesalahan dalam hasil analisis. Oleh karena itu, penting untuk mengidentifikasi di mana nilai-nilai tersebut hilang dan memutuskan bagaimana cara menangani mereka.
Dengan menggunakan bantuan metode isna() yang dijumlahkan, hasil terlihat bahwa terdapat missing value pada kolom Engine Fuel Type sebayak 3 sel, Engine HP sebanyak 69 sel, Engine Cylinders 30 sebanyak 30 sel, Number of Doors sebanyak 6 sel, dan Market Category sebanyak 3742 sel.
Berikut visualisasi dengan bantuan library seaborn pada Gambar 1.
![Gambar 1](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/image.png).

  Berdasarkan Gambar 1, terlihat jelas pada bintik-bintik yang linear dengan nama-kolom mengandung missing value. 
* Checking the duplicate rows
Tahap ini bertujuan untuk melihat apakah ada baris data yang identik atau duplikat dalam dataset? Duplikasi data bisa menjadi masalah karena mereka dapat mempengaruhi hasil analisis statistik dengan memberikan bobot tambahan pada observasi yang sama. Mengidentifikasi dan menghapus duplikasi dapat membantu
memastikan keakuratan analisis data dan mencegah distorsi dalam hasil.
Pada notebook, proses ini menggunakan beberapa metode dari library pandas seperti shape untuk melihat ukuran data dari baris dan kolom, duplicated() untuk melihat data-data yang terduplikasi, dan count() untuk menghitung keseluruhan data dalam baris.
Hasil/output yang muncul didapat kesimpulan bahwa jumlah data yang terduplikasi adalah 715 data dari seluruh kolom yang berjumlah 16.
Kemudian, visualisasi jumlah data dari setiap kolom digunakan untuk melihat perbedaan sebelum kolom yang terduplikasi dihapus dan sebagai bukti berhasil atau tidaknya proses tersebut.

* Detecting Outliers
Outliers adalah nilai yang menonjol secara statistik berbeda dari mayoritas nilai dalam dataset. Mereka dapat menyebabkan bias dalam analisis statistik atau model yang dibangun dari data tersebut. Outliers juga dapat mengakibatkan testing model menghasilkan overfitting atau underfitting. Maka dari itu, proses ini sangat dibutuhkan.
Dengan menggunakan library seaborn dengan methodnya boxplot(), parameter di dalamnya dapat disi dengan kolom numerik dari dataset yang digunakan.
Gambar 2. Salah satu contoh visualisasi keberadaan outlier dari kolom Engine HP.
![Gambar 2](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/outliers.png)

   Berdasarskan visualisasi pada gambar 2, terlihat bahwa terdapat beberapa fitur numerik yang mengandung outlier. Nantinya, Outlier-outlier tersebut akan dihapus menggunakan teknik atau metode IQR. 

### Data Cleaning
Data Cleaning adalah proses pembersihan data yang bertujuan untuk memastikan kualitas dan konsistensi data sebelum dilakukan analisis lebih lanjut. Tujuan utamanya adalah untuk menghilangkan masalah atau gangguan dalam dataset yang dapat memengaruhi hasil analisis statistik atau pembangunan model. Pada tahap ini akan dilakukan beberapa proses seperti Renaming the columns, Dropping the missing or null values, Dropping the duplicated rows, dan Handling the outliers. Berikut implemetasi dan hasil yang didapat:
* Renaming the columns:
Kenapa proses ini diperlukan? karena pada awalnya, penolahan dataset tidak selamanya melakukan renaming kolom-kolom dengan bahasa yang pasti sesuai dengan kebutuhan bisnis. Maksudnya, penamaan kolom-kolom tersebut bisa jadi mengikuti aturan tertentu sehingga terkesan umum. Meskipun bersifat opsional, proses tersebut dibutuhkan dalam project ini utnuk mengubah kolom MRSP menjadi Harga Jual.
* Dropping the missing or null values
Berasarkan tahap sebelumnya, teradapat beberapa missing value yang dapat mengganggu proses analisis dan pembuatan prediktif model. Maka perlu adanya tindakan lebih lanjut untuk menangani keberadaanya, yaitu menghapus missing value. Teknik yang akan digunakan adalah metode dropna() dari library pandas. Hasilnya kolom-kolom yang bermasalah seperti Engine Fuel Type sebayak 3 sel, Engine HP sebanyak 69 sel, Engine Cylinders 30 sebanyak 30 sel, Number of Doors sebanyak 6 sel, dan Market Category sebanyak 3742 sel berhasil dibersihkan.
Gambar 3. Gambar untuk menunjukan perbedaan dari tahap sebelumnya pada data assesing.
![Gambar 3](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/clean.png)

  Berdasarkan tersebut, perbedaan yang dapat dibandingkan dengan Gambar 2 sebelumnya bahwa Gambar 3 terlihat bersih.
* Dropping the duplicated rows
Terlihat di tahap sebelumnya bahwa terdapat data yang terduplikasi. Kali ini, proses lanjutan akan dilakukan untuk mengurangi gangguan yang ada pada data karena jumlahnya yang besar. Dengan menggunakan metode drop_duplicates() dari library pandas ke dataset ini proses menghapus duplikasi data dalam baris berhasil dilakukan. Bukti menunjukan bahwa jumlah data dalam baris berkurang dari yang awalnya 11914 menjadi 7735.
*  Handling the outliers
Berdasarkan visualisasi keberadaan outlier di tahap sebelumnya, proses ini penting untuk dilakukan agar akurasi model yang dilatih tidak terpengaruh secara signifikan. Teknik yang akan digunakan seperti yang sudah disinggung sebelumnya, yaitu IQR. Teknik ini mengidentifikasi outlier yang ada dibatas atas Q3 dan dibatas bawah atau Q1. Lalu, nilai-nilai yang ada di dalam batas akan digunakan sementara yang diluar kedua batas atau outlier akan dihapus. Formula yang lebih jelas untuk IQR sebagai berikut:
$Batas\ bawah = Q1 - 1.5 * IQR$ dan $Batas\ atas = Q3 + 1.5 * IQR$.
Hasil dapat terlihat dari pengurangan jumlah data dalam baris yang sebelumnya berjumlah 7735 menjadi 5622 dalam bentuk tabel sederhana dengan bantuan metode info().
Gambar 4. Hasil final setelah proses cleaning data.

![Gambar 4](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/Screenshot%20(247).png).
### Exploratory Data Analysis -Univariate Analysis
Kenapa Univariate Analysis? jadi tujuan melakukan analysis ini untuk memahami karakteristik dari satu variabel tunggal dalam dataset tanpa ada memperhatikan hubungan variabel lain. Tahap kali ini, Univariate analysis dilakukan dengan memisah categorical feature dengan numerical features.
* Categorical Features (Fitur Kategorik)
Proses awal pemisahan kedua fitur menggunakan bantuan metode select_dtypes dengan parameter include untuk categorical features. Lalu, cara memvvisualisasikan fitur ini secara univariate dapat dilakukan dengan bantuan metode histogram dari library plotly. Berikut ini adalah salah satu gambar yang dapat dijadikan sampel output.
![Gambar 5](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/newplot.png)

  Gambar 5. Visualisasi univariate pada kolom Make atau Merek.
  Berdasarkan gambar tersebut, TOp 5 merek mobil yang paling banyak jumlahnya, yaitu Chevrolet, Volkswagen, Cadillac, Infiniti, dan Nissan.
                
* Numerical Features (Fitur Numerik)
Pada Numerical Features, proses pemisahan menggunakan metode select_dtypes dengan parameter exclude untuk melakukan pemilihan tipe data. Cara memvvisualisasikan fitur ini secara univariate dapat dilakukan dengan metode histogram dari library plotly. Berikut ini adalah visualisasi univariate dari kolom-kolom numerik.
![Gambar 6](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/multivariate.png)

  Gambar 6. Visualisasi univariate analisis pada kolom numerik.
  Berdasarkan gambar tersebut, histogram pada variabel "Harga_Jual", "highwat MPG", "Engine HP" dan "city mpg" memiliki beberapa karakteristik seperti:
* Peningkatan harga jual mobil terdistribusi dengan cukup baik. Hal ini dapat dilihat pada histogram dari keempat kolom tersebut yang mana sampel cenderung meningkat lalu mengalami penurunan seiring dengan meningkatnya harga jual rumah.
* Distribusi harga cenderung cukup normal. Hal ini kemungkinan besar akan berimplikasi pada model.
### Exploratory Data Analysis -Multivariate Analysis
Multivariate Analysis adalah analisis yang memperlihatkan korelasi atau hubungan dua atau lebih variabel. Analysis ini bertujuan untuk memahami hubungan antara dua atau lebih variabel dalam sebuah dataset. Seperti analysis sebelumnya, kali ini juga melakukan pemisahan antara numerical feature dan categorical feature. Nantinya, masing masing feature tersebut akan dianalisis terhadap variabel target, yaitu Harga_Jual.
* Categorical Features
Cara memvvisualisasikan fitur ini secara multivariate dapat dilakukan dengan bantuan metode histogram dari library plotly. Bedanya dengan analisis univariate adalah kali ini kita memasukan dua parameter yang diperlukan, yaitu x yang berisi beberapa variabel independent dan y yang berisi variabel dependent atau target (Harga_Jual). Berikut ini adalah salah satu gambar yang dapat dijadikan sampel output.
![Gambar 7](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/newplot%20(2).png).

  Gambar 7. Visualisasi multivariate Analisis pada kolom Veichle Size.
  Berdasarkan gambar tersebut dari fitur veichle Size terlihat bahwa pengaruh terhadap terhadap variabel target atau Harga_Jual bervariasi. Misalkan, pada bagian "Midsize" terlihat memiliki korelasi tertinggi dengan variabel target (Harga_Jual), sementara yang lain relatif sama.
* Numerical Features
Untuk mengamati hubungan antara fitur numerik, kita akan menggunakan fungsi pairplot(). Kita juga akan mengobservasi korelasi antara fitur numerik dengan fitur target menggunakan fungsi corr(). Berikut adalah gambar dari kedua cara tersebut.
Gambar 8. Visualisasi multivariate pada numerical features dengan fungsi pairplot().
![Gambar 8](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/real%20multivariate.png).

  Gambar 9. Visualisasi multivariate pada numerical features dengan fungsi corr().
![Gambar 9](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/cmaps.png).

  Kesimpulan yang bisa diambil berdasarkan kedua teknik tersebut pada numerical feature sebagai berikut:
* Kolom yang memiliki korelasi tertinggi dengan variabel target(Harga_Jual) adalah Engine Hp dengan skor korelasi diatas 70%.
* Dapat disimpulkan bahwa kolom-kolom yang berkorelasi sedang dengan variabel target Harga_Jual adalah higway MPG, city mpg, dan Engine Cylinders.
* Dapat disimpulkan bahwa kolom-kolom yang berkorelasi rendah dengan variabel target Harga_Jual adalah Popularity, Year, dan Number of Doors. Lalu, kolom-kolom yang memiliki korelasi rendah ini nantinya akan dihapus karena dapat mempengaruhi kinerja model dalam memprediksi variabel target Harga_Jual. Dengan menghapus kolom-kolom yang memiliki korelasi rendah,  teknik dapat mengurangi dimensi fitur dan meningkatkan performa model dapat dilakukan dalam proses pembelajaran dan prediksi.
## Data Preparation
Data preparation merupakan tahapan penting dalam proses pengembangan model machine learning. Ini adalah tahap di mana  proses transformasi pada data akan dilakukan dengan menjadikan data ke bentuk yang cocok untuk proses pemodelan. Terdapat beberapa tahapan di dalamnya Encoding fitur kategori, Reduksi dimensi dengan PCA, proporsi dataset dengan fungsi train_test_split, dan Standarisasi.
### Encoding Feature Category 
Ini adalah proses mengubah fitur kategorikal atau kualitatif menjadi representasi numerik yang dapat dimengerti oleh algoritma machine learning. Fitur kategorikal adalah fitur yang memiliki nilai dalam kategori atau kelompok tertentu tanpa urutan yang terdefinisi. Proses ini menggunakan teknik one-hot-encoding dengan tujuan untuk mendapatkan fitur baru yang sesuai. Fitur baru ini nantinya dapat digunakan untuk mewakili fitur kategori. Untuk teknik one-hot encoding ini, metode yang dibutuhkan adalah concat dan get_dummies dari library Scikit-learn. Berikut adalah tabel sebagai hasil/output dari proses diatas:
Tabel 2. Hasil dari proses Encoding Feature Category.
|index|Engine HP|Engine Cylinders|highway MPG|city mpg|Harga\_Jual|Make\_Acura|Make\_Alfa Romeo|Make\_Audi|Make\_BMW|Make\_Buick|Make\_Cadillac|Make\_Chevrolet|Make\_Chrysler|Make\_Dodge|Make\_FIAT|Make\_GMC|Make\_Genesis|Make\_HUMMER|Make\_Honda|Make\_Hyundai|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|335\.0|6\.0|26|19|46135|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|
|1|300\.0|6\.0|28|19|40650|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|
|2|300\.0|6\.0|28|20|36350|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|
|3|230\.0|6\.0|28|18|29450|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|
|4|230\.0|6\.0|28|18|34500|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|
### Reduksi Dimensi dengan PCA
Teknik reduksi (pengurangan) dimensi adalah prosedur yang mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data. Teknik pengurangan dimensi yang paling populer adalah Principal Component Analysis atau disingkat menjadi PCA. Ia adalah teknik untuk mereduksi dimensi, mengekstraksi fitur, dan mentransformasi data dari “n-dimensional space” ke dalam sistem berkoordinat baru dengan dimensi m, di mana m lebih kecil dari n.  Hal ini sangat penting karena jika dataset memiliki banyak fitur yang mempersulit analisis, hal tersebut dapat menyebabkan overfitting atau masalah lain.

PCA bekerja menggunakan metode aljabar linier. Teknik ini mengasumsikan bahwa sekumpulan data pada arah dengan varians terbesar merupakan yang paling penting (utama). PCA umumnya digunakan ketika fitur dalam data memiliki korelasi yang tinggi. Korelasi tinggi ini menunjukkan data yang berulang atau redundant. 

Untuk mengetahui fitur-fitur yang memiliki korelasi tinggi maka dapat didasarkan atas visualisasi dengan fungsi pairplot(). Gambar 8 yang telah ditunjukan sebelumnya akan dijadikan acuan analisis dalam visualisasi ini untuk menentukan fitur mana yang cocok digunakan. Pada gambar tersebut fitur "highway MPG" terhadap fitur "city mpg" memiliki korelasi yang cukup baik dan mengandung informasi yang sama, yaitu luas distribusi berdasarkan fungsi pairplot().

![Gambar 10](https://github.com/Symphoen1x/Dicoding_Expert/blob/main/pairplot%20keren.png).

  Selanjutnya akan dilakukan proses memanggil class PCA() dari library scikit-learn. Paremeter yang akan  dimasukkan ke dalam class adalah n_components dan random_state. Parameter n_components merupakan jumlah komponen atau dimensi, berdasarkan Gambar 10  maka jumlah yang dimasukan ada 2. Kemudian akan muncul hasil berupa array yang merupaakn Principal Component (PC). PC pertama maksudnya informasi mengenai dua fitur diatas sebagian besar terdapat di PC tersebut, sementara sisanya terdapat di PC dua. Pc pertama ini mewakili dua fitur melalui sebuah variabel baru yang bernama "dimension". Berikut dataframe hasil proses PCA.
Tabel 3. Dataframe hasil akhri dari proses PCA
|index|Engine HP|Engine Cylinders|Harga\_Jual|Make\_Acura|Make\_Alfa Romeo|Make\_Audi|Make\_BMW|Make\_Buick|Make\_Cadillac|Make\_Chevrolet|Make\_Chrysler|Make\_Dodge|Make\_FIAT|Make\_GMC|Make\_Genesis|Make\_HUMMER|Make\_Honda|Make\_Hyundai|Make\_Infiniti|Make\_Kia|dimension|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|--|
|0|335\.0|6\.0|46135|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|-1.117210|
|1|300\.0|6\.0|40650|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0.452164|
|2|300\.0|6\.0|36350|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|1.072056|
### Train-Test-Split
Pada awalnya, proses scaling pada seluruh dataset membuat model memiliki informasi mengenai distribusi pada data uji. Informasi tentang data uji (yang seharusnya tidak dilihat oleh model) turut diikutsertakan dalam proses transformasi data latih. Oleh karena itu, kita akan melakukan proses scaling secara terpisah antara data latih dan data uji. Pada tahap kali ini proses pembagian data menjadi data latih dan data uji dengan proporsi 80:20. Hasil yang muncul dari pembagian data diatas, train data sejumlah 4497 dan test data sejumlah 1125 dari total 5622 sampel data. Proses ini dibantu oleh library sklearn dengan modul train_test_split.
### Standarisasi
Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Teknik ini digunakan untuk transformasi pada feature numerik. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Untuk menghindari kebocoran informasi pada data uji, penerapan dilakukan untuk fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, standarisasi dilakukan pada data uji. 
Tabel 4. Dataframe hasil akhir setelah proses standarisasi pada fitur numerik.
|index|Engine HP|Engine Cylinders|dimension|
|---|---|---|---|
|3465|-0\.623736691170428|0\.36597140250540167|-1\.6271285525468915|
|5206|-0\.03471025265858132|0\.36597140250540167|-0\.4812240283983712|
|11487|-0\.3932480847962271|-1\.0305270252135548|0\.9545158750588465|
|10798|-1\.0462991361897962|-1\.0305270252135548|0\.922708673076368|
|1020|-1\.1359335942242075|-1\.0305270252135548|1\.4180144445216696|

  Berdasarkan output diatas, terlihat benar bukan bahwa standarisasi mengubah nilai mean menjadi 0 dan nilai standar devisiasi menjadi 1. Sekitar 68% dari nilai akan berada di antara -1 dan 1.


 TO-do:
* Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.

* Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. Jelaskan proses improvement yang dilakukan.

* Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. Jelaskan mengapa memilih model tersebut sebagai model terbaik.

  
## Model Devlopment
Proses kali ini menunjukan penggunaan machine learning dengan beberapa Algoritma yang akan digunakan. Algoritma yang akan digunakan pada proses Model Devploment kali ini ada tiga, yaitu K-Nearest Neighbor, Random Forest, dan Boosting Algorithm. Sebelum itu, pembuatan DataFrame yang berisi ketiga algoritma diatas untuk membandingkan hasil prediksi terbaik perlu dibuat. Tidak lupa untuk menggunakan hyperparameter tuning [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) dalam proses pelatihan model dengan kaetiga algoritma tersebut. Tujuanya untuk mengingkatkan performa model yang dilatih. Parameter ini nantinya dibutuhkan dalam mencari parameter terbaik dari masing masing algoritma yang dilakukan pelatihan.
### Model Devlopment Menggunakan Algoritma K-Nearest Neighbor atau KNN
[KNN](https://www.ibm.com/topics/knn#:~:text=Next%20steps-,K-Nearest%20Neighbors%20Algorithm,of%20an%20individual%20data%20point) adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN ini cocok digunakan untuk kasus regresi dan klasifikasi dalam machine learning.
Algoritma K-Nearest Neighbor (KNN) bekerja dengan cara mencari K tetangga terdekat data input baru berdasarkan jaraknya, lalu memprediksi output berdasarkan rata-rata output tetangga tersebut. 
Kelebihan KNN:

* Sederhana untuk diimplementasikan dan dimengerti.
* Tidak memerlukan pelatihan model secara eksplisit.
* Fleksibel dan dapat menangani data non-linear.
  
Kekurangan KNN:

* Sensitif terhadap noisy data dan outlier.
* Membutuhkan banyak memori untuk menyimpan seluruh data latih.
* Prediksi dapat menjadi lambat karena perhitungan jarak untuk data latih yang besar.
* Hasil sangat tergantung pada pemilihan nilai K dan fungsi jarak yang digunakan.
  
Kali ini, modelling dengan Algoritma KNN akan menggunakan bantuan hyperparameter tuning untuk menemukan kombinasi nilai optimal untuk hyperparameter dari sebuah model machine learning dengan tujuan meningkatkan performa model. Proses awal penggunaan algortima ini dengan memanggil metode [KNeighborsRegressor()](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) dari library scikit-learn. Kemudian parameter yang digunakan di algoritma ini, yaitu n_neigbors akan di sesuaikan berdasarkan hyperparameter tunning. Output yang muncul dari proses tersebut adalah skor RandomSearch untuk KNN sebesar -23871881.20319546 dan parameter n_neigbors = 10. Kemudian, Masuk ke proses pelatihan model KNN dengan parameter n_neigbors menggunakan hasil dari tahapan sebelumnya.    ditrain oleh fungsi fit() dengan matrix evaluasi yang digunakan adalah Mean Squared Error atau MSE. Hasil yang terlihat 

### Model Devlopment Menggunakan Algoritma Random Forest
Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama.

Kali ini, modelling dengan Algoritma Random Forest akan menggunakan bantuan hyperparameter tuning untuk menemukan kombinasi nilai optimal untuk hyperparameter dari sebuah model machine learning dengan tujuan meningkatkan performa model.
### Model Devlopment Menggunakan Algoritma Boosting Algorithm
Algoritme boosting menggabungkan beberapa pembelajar lemah dalam metode berurutan, yang secara iteratif. Pendekatan ini membantu mengurangi bias tinggi yang umum terjadi pada model machine learning. Cara kerjanya adalah membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.

Kali ini, modelling dengan Boosting Algorithm akan menggunakan bantuan hyperparameter tuning untuk menemukan kombinasi nilai optimal untuk hyperparameter dari sebuah model machine learning dengan tujuan meningkatkan performa model.
## Evaluasi Model
Tahap evaluasi dalam membangun model bertujuan untuk mengukur kinerja dan keefektifan model yang telah dibuat. Evaluasi model penting karena memberikan pemahaman tentang seberapa baik model dapat melakukan prediksi atau menggeneralisasi data baru yang tidak terlihat selama pelatihan. Pada tahap ini, penggunaan metrik akan dilakukan. Metrik yang akan digunakan adalah Mean Squared Eror atau MSE. Metrik ini menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi melalui sebuah persamaan.

Jadi, proses dalam tahap evaluasi adalah scaling numeric features, count MSE for data train and test, plot metrix with bar chart, prediction target variable, Calculates the difference between the predicted value and the y_true value, model accuracy with difference algorithm, and
### Scaling numeric features
### Count MSE for data train and test
### Plot metrix with bar chart
### Prediction target variable
### Calculates the difference between the predicted value and the y_true value
### Accuracy Model with difference algorithm

## Conclusion
