# -*- coding: utf-8 -*-
"""Submit_project1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xnQKzqBOHIDajcPH-iiEgnvEJw0ilMI6

# Project Predictive Analytics By Isa Aulia Almadani - Prediksi Harga Jual Mobil dengan Machine Learning
* Proyek ini bertujuan untuk membuat model prediksi harga jual mobil berdasarkan fitur-fitur utama yang ada pada mobil.
* Proyek ini dikerjakan untuk memenuhi persyaratan menyelesaikan kelas [Machine Learning Terapan by Dicoding](https://www.dicoding.com/academies/319/tutorials/17052)
* Nama: Isa Aulia Almadani
* Asal Domisili: Sukoharjo, Jawa Tengah

## **Dara Understanding**
Ini adalah bagian awal dari alur pembuatan model. Tahapanya ada import library yang diperlukan, pengumpulan data yang terdapat di github, dan pemuatan dataset CSV dari GitHUbke dalam DataFrame melalui Pandas.

### 1. Importing the required libraries for model
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.preprocessing import  OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

"""### 2. Data Gathering and Data Loading"""

url = 'https://github.com/Symphoen1x/Dicoding_Expert/blob/main/car_new.csv?raw=true'
df_early= pd.read_csv(url)
df_early

"""### 3. Description variable

"""

df_early.info()

for col in df_early:
    print(col)
    print(df_early[col].unique())
    print(df_early[col].nunique())
    print(20*'=')

"""**Notes:**

Variabel - variabel pada dataset [Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset) diantaranya:
* Make: Merek/manufaktur mobil (BMW, Acura, linkoln, dll).
*	Model: Model spesifik dari mobil (1 Series M, 1 Series, dll).
*	Year: Tahun produksi mobil.
*	Engine Fuel Type: Jenis bahan bakar mesin (gasoline, diesel, etc).
*	Engine HP: Tenaga kuda mesin dalam satuan horsepower.
*	Engine Cylinders: Jumlah silinder mesin.
*	Transmission: Tipe transmisi (manual, auto, dll).
* Type	Driven_Wheels:Roda penggerak (rear wheel drive, all wheel drive	, dll).
*	Number of Doors: Jumlah pintu mobil.
*	Market Category: Kategori pasar mobil (Factory Tuner,Luxury,High-Performance, Luxury, dll).
* Vehicle Size: Ukuran mobil (Compact, midsize, dll).
* Vehicle Style: Gaya/model mobil (Coupe, Convertible, dll).
* highway MPG: Jarak tempuh per galon di jalan tol.
* city mpg: Jarak tempuh per galon di perkotaan.
* Popularity: Tingkat popularitas mobil.
* MSRP: Harga jual yang disarankan produsen.

Untuk informasi detailnya terdapat dalam laporan project ini.

**Notes:**
Berdasarkan visualisasi DataFrame tersebut, informasi di dalamnya dapat lihat sebagai berikut:
* Dataset ini memiliki 11.913 baris dan 16 kolom. Sumber dataset: [Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset).
* Masing masing variabel memiliki nilai unik yang bervariasi.

## Exploratory Data Analysis or (EDA)
EDA merupakan proses fundamental dalam analytics untuk memahami dan mengeksplorasi data sebelum melakukan modelling. EDA sangat diperlukan dalam analisis karena membantu menghasilkan model yang akurat. Pada proses EDA kali ini, terdapat lima tahapan yang akan dilakukan, yaitu Check characteristic data, Data Assesing, Data Cleaning, Univariate Analysis, dan Multivariate Analysis.

### 1. Check characteristic data
"""

df_early.describe()

"""### 2. Data Assesing
Tahap assesing/penilaian bertujuan untuk menganalisis bagaimana kualiatas seluruh data yang ada dalam kedua dataset tersebut seperti missing value yang ada, keberadaan duplikasi data yang ada, keberadaan outlier dari kedua dataset, perlukah mengganti nama variabel sesuai tujuan, kesesuaian tipe variabel, dll.

#### 2.1 Checking the missing or null values
Proses ini penting dilakukan karena fitur-fitur memiliki nilai null/NaN dapat mempengaruhi hasil analisis data.
"""

print(df_early.isna().sum())

sns.heatmap(df_early.isna(), cmap='viridis')

"""**Notes:**

* Berdasarkan output di atas, terdapat lima fitur yang memiliki nilai null/NaN. Fitur-fitur tersebut nantinya akan dihapus di tahap Data Cleaning.
* Salah satu visualisasi di atas akan dijadikan bukti keberhasilan proses penanganan masalah ini.

#### 2.2 Checking the duplicate rows
Tujuan dari proses ini untuk melihat keberadaan data yang terduplikasi
"""

df_early.shape

detactor =df_early[df_early.duplicated()]
print(f'jumlah data terduplikasi, jumlah kolom: {detactor.shape}')

df_early.count()

"""**Notes:**
* Berdasarkan visualisasi di atas terlihat bahwa jumlah data yang terduplikasi adalah 715 data dari seluruh kolom yang berjumlah 16.
* Kemudian, visualisasi jumlah data dari setiap kolom digunakan untuk melihat perbedaan sebelum kolom yang terduplikasi dihapus dan sebagai bukti berhasil atau tidaknya proses tersebut.

#### 2.3 Detecting Outliers
Tujuan dari tahap ini untuk melihat keberadaan dari outliers pada data.
"""

df_early.info()

sns.boxplot(x=df_early['Year'])

sns.boxplot(x=df_early['Engine HP'])

sns.boxplot(x=df_early['Engine Cylinders'])

sns.boxplot(x=df_early['highway MPG'])

sns.boxplot(x=df_early['city mpg'])

sns.boxplot(x=df_early['Popularity'])

"""**Notes:**
* Berdasarskan visualisasi untuk mengecek keberadaan outlier diatas, terlihat bahwa terdapat beberapa fitur numerik yang mengandung outlier. Nantinya, Outlier-outlier tersebut akan dihapus menggunakan teknik atau metode IQR.

### 3. Data Cleaning
Data Cleaning adalah proses pembersihan data yang bertujuan untuk memastikan kualitas dan konsistensi data sebelum dilakukan analisis lebih lanjut. Tujuan utamanya adalah untuk menghilangkan masalah atau gangguan dalam dataset yang dapat memengaruhi hasil analisis statistik atau pembangunan model. Pada tahap ini akan dilakukan beberapa proses seperti Renaming the columns, dan Dropping the duplicated rows, Handling the outliers. Berikut implemetasi dan hasil yang didapat:

#### 3.1 Renaming the columns
Tujuanya agar nama variabel yang akan digunakan sesuai dengan yang ada di problem statements.
"""

df_early = df_early.rename(columns={'MSRP' : 'Harga_Jual'})
df_early.head()

"""**Notes:**
* Proses ini mengacu pada kolom target, yaitu MRSP untuk diubah menjadi Harga_Jual.

#### 3.2 Dropping the missing or null values
Berasarkan tahap sebelumnya, teradapat beberapa missing value yang dapat mengganggu proses analisis dan pembuatan prediktif model. Maka perlu adanya tindakan lebih lanjut untuk menangani keberadaanya, yaitu menghapus missing value.
"""

df_early = df_early.dropna()
print(df_early.isnull().sum())

sns.heatmap(df_early.isna(), cmap='viridis')

"""**Notes:**
* Seperti yang sudah dijelaskan sebelumnya bahwa untuk pembuktian bagaimana proses ini berhasil dilakukan atau tidak, visualisasi jumlah data di masing-masing kolom kembali di perlihatkan setelah proses penghapusan missing atau null value. Hasilnya, jumlah data telah berkurang dengan jumlah 8084 disetiap kolom dan missing atau null values sudah hilang.

#### 3.3 Dropping the duplicate rows
Terlihat di tahap sebelumnya bahwa terdapat data yang terduplikasi. Kali ini, proses lanjutan akan dilakukan untuk mengurangi gangguan yang ada pada data karena jumlahnya yang besar.
"""

df_early = df_early.drop_duplicates()
df_early.head()

df_early.count()

"""**Notes:**
* Setelah proses penanganan dilakukan, masalah duplikasi data dalam baris sudah berhasil di hapus sehingga jumlah data pun ikut berkurang di setiap kolom.

#### 3.4 Handling the outliers
Berdasarkan visualisasi keberadaan outlier di tahap sebelumnya, proses ini penting untuk dilakukan agar akurasi model yang dilatih tidak terpengaruh secara signifikan. Teknik yang akan digunakan seperti yang sudah disinggung sebelumnya, yaitu IQR. Teknik ini mengidentifikasi outlier yang ada dibatas  atas Q3 dan dibatas bawah atau Q1. Lalu, nilai-nilai yang ada di dalam batas akan digunakan sementara yang diluar kedua batas atau outlier akan dihapus
"""

Q1 = df_early.quantile(0.25)
Q3 = df_early.quantile(0.75)
IQR=Q3-Q1
df_final = df_early[~((df_early<(Q1-1.5*IQR))|(df_early>(Q3+1.5*IQR))).any(axis=1)]

df_final.shape

df_final.count()

df_final.info()

"""**Notes:**
* Berdasarkan hasil dari teknik diatas bahwa visualisasi menunjukana adanya perbedaan jumlah data yang ada disetiap kolom. Itu menandakan bahwa dataset kita telah bersih dan menyisakan 5622 sampel.

### 4. Exploratory Data Analysis -Univariate Analysis
Kenapa Univariate Analysis? jadi tujuan melakukan analysis ini untuk memahami karakteristik dari satu variabel tunggal dalam dataset tanpa ada memperhatikan hubungan variabel lain. Tahap kali ini, Univariate analysis dilakukan dengan memisah categorical feature dengan numerical features.

#### 4.1 Separate the numerical and categorical features
"""

numerical_features = df_final.select_dtypes(include = [np.number])
categorical_features = df_final.select_dtypes(exclude = [np.number])

"""**Notes:**
* Proses ini dilakukan dengan bantuan library numpy beserta parameter yang diperlukan.
* Include mengacu pada tipe data numerik, sementara exclude sebaliknya kecuali tipe data numerik atau mengacu pada tipe data kategorik.

#### 4.1 Categorical Feature
"""

for var in categorical_features:
  count = df_final[var].value_counts()
  percent = 100 * df_final[var].value_counts(normalize=True)
  df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
  print(f"Statistik untuk fitur {var}:\n")
  print(df)
  print("\n")
  fig = px.histogram(categorical_features, x=var, color_discrete_sequence = ['#C9A26B'])
  fig.show()

"""**Notes:**
* Berdasarkan visualisasi diatas, muncul karakteristik data untuk setiap kolom kategorikal dengan jumlah setiap datanya.
* Terlihat bahwa setiap kolom kategorikal diatas memiliki karakteristik yang unik dan berbeda-beda

#### 4.2 Numerical Features
"""

df_final.hist(bins=50, figsize=(20,15))
plt.show()

"""**Notes:**
* Sebagai target atau y_true, variabel Harga_Jual terdistribusi dengan cukup baik karena terlihat sampel data cenderung meningkat di awal kemudian semakin turun sejalan  dengan peningkatan tenaga mesin, jumlah pintu, dan MPG.
* Tenaga mesin dan silinder mesin umumnya meningkat selama bertahun-tahun. Hal ini kemungkinan disebabkan oleh meningkatnya permintaan akan kinerja dan efisiensi bahan bakar.

### 5. Exploratory Data Analysis -Multivariate Analysis
Analysis ini bertujuan untuk memahami hubungan antara dua atau lebih variabel dalam sebuah data set. Seperti analysis sebelumnya, kali ini juga melakukan pemisahan antara numerical feature dan categorical feature. Nantinya, masing masing feature tersebut akan dianalisis terhadap variabel target, yaitu Harga_Jual.

#### 5.1 Categorical Features
"""

for var in categorical_features:
    fig = px.histogram(df_final, x=var, y='Harga_Jual', color_discrete_sequence=['#C9A26B'])
    fig.update_layout(title="Rata-rata CNT Relatif terhadap - {}".format(var))
    fig.show()

"""**Notes:**
* Berdasarkan visualisasi diatas, insight yang bisa diperoleh sangat banyak dari setiap hubungan variabel-variabel numerik terhadap Harga_Jual.
* Beberapa ada yang berpengaruh tinggi seperti di fitur Transmission Type, Harga_Jual memiliki pengaruh paling tinggi di tipe Automatic.
* Beberapa diantaranya juga sangat rendah seperti fitur Veichel Style, terlihat bahwa 2dr SUV memiliki pengaruh yang kecil terhadap variabel target atau Harga_Jual.
* Untuk informasi mengenai visualisasi secara spesifik dari setiap variabel akan diperjelas di file laporan.

#### 5.2 Numerical Features
"""

sns.pairplot(df_final, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
c = df_final.corr().round(2)

# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=c, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

df_final.head()

"""**Notes:**
* Berdasarkan visualisasi diatas, terdapat dua jenis visualisasi untuk fitur-fitur numerik, yaitu dengan bantuan Pairplot dan Corellation Matrix.
* Kolom yang memiliki korelasi tertinggi dengan variabel target Harga_Jual adalah Engine Hp.
* Dapat disimpulkan bahwa kolom-kolom yang berkorelasi sedang dengan variabel target Harga_Jual adalah higway MPG, city mpg, dan Engine Cylinders.
* Dapat disimpulkan bahwa kolom-kolom yang berkorelasi rendah dengan variabel target Harga_Jual adalah Popularity, Year, dan Number of Doors.
* Untuk kolom-kolom yang memiliki korelasi rendah maka akan dihapus.

#### 5.3 Dropping Irrelevant Columns
Berdasarkan hasil dari analisis diatas menunjukan bahwa terdapat beberapa fitur yang berhubungan kuat dengan fitur harga jual dan terdapat pula yang benar-benar lemah dengan indikasi nilai yang kecil seperti 0.02 dan lain-lain. Maka perlu untuk dilakukan penghapusan kolom sebagai solusi agar tujuan awal terpenuhi.
"""

df_final.drop(['Number of Doors', 'Popularity', 'Year'], inplace=True, axis=1)
df_final.head()

df_final.describe()

"""## Data Preparation
Tahapan kali ini dilakukan dengan tujuan mengubah bentuk data menjadi bentuk yang cocok untuk proses pemodelan. Terdapat beberapa tahapan di dalamnya Encoding fitur kategori, Reduksi dimensi dengan PCA, proporsi dataset dengan fungsi train_test_split, dan Standarisasi.

### 1. Encoding Feature Category
Tahapan ini menggunakan teknik one-hot-encoding dengan tujuan untuk mendapatkan fitur baru yang sesuai. Fitur baru ini nantinya dapat digunakan untuk mewakili variabel kategori.
"""

df_final = pd.concat([df_final, pd.get_dummies(df_final['Make'], prefix='Make')],axis=1)
df_final = pd.concat([df_final, pd.get_dummies(df_final['Engine Fuel Type'], prefix='Engine Fuel Type')],axis=1)
df_final = pd.concat([df_final, pd.get_dummies(df_final['Market Category'], prefix='Market Category')],axis=1)
df_final = pd.concat([df_final, pd.get_dummies(df_final['Vehicle Style'], prefix='Vehicle Style')],axis=1)
df_final = pd.concat([df_final, pd.get_dummies(df_final['Vehicle Size'], prefix='Vehicle Size')],axis=1)
df_final = pd.concat([df_final, pd.get_dummies(df_final['Model'], prefix='Vehicle Size')],axis=1)



df_final = pd.concat([df_final, pd.get_dummies(df_final['Driven_Wheels'], prefix='Driven_Wheels')],axis=1)
df_final = pd.concat([df_final, pd.get_dummies(df_final['Transmission Type'], prefix='Transmission Type')],axis=1)
df_final.drop(['Model', 'Market Category', 'Engine Fuel Type', 'Vehicle Style', 'Vehicle Size', 'Engine Fuel Type', 'Make','Driven_Wheels','Transmission Type'], axis=1, inplace=True)
df_final.head()

"""### 2. Reduksi Dimensi dengan PCA
Tahapan kali ini menggunakan teknik pengurangan dimensi yang bernama Principal Component Analysis (PCA). Tujuan dilakukan teknik ini untuk mengurangi dimensi dari data yang berdimensi tinggi atau dengan kata lain mengubah representasi data dengan banyak variabel menjadi data dengan jumlah variabel yang lebih sedikit. Hal ini sangat penting karena jika dataset memiliki banyak fitur yang mempersulit analisis, hal tersebut dapat menyebabkan overfitting atau masalah lain.
"""

sns.pairplot(df_final[['highway MPG','city mpg']], plot_kws={"s":2})

pca = PCA(n_components=2, random_state=123)
pca.fit(df_final[['highway MPG','city mpg']])
princ_comp = pca.transform(df_final[['highway MPG','city mpg']])

pca.explained_variance_ratio_.round(2)

pca = PCA(n_components=1, random_state=123)
pca.fit(df_final[['highway MPG','city mpg']])
df_final['dimension'] = pca.transform(df_final.loc[:, ('highway MPG','city mpg')]).flatten()
df_final.drop(['highway MPG','city mpg'], axis=1, inplace=True)

df_final.head(3)

df_final.info()

"""**Notes:**
* Berdasarkan visualisasi diatas, fitur numerik yang digunakan dalam proses PCA tersebut ada dua fitur, yaitu highway MPG dan	city mpg. Kenapa tahapan ini menggunakan kedua fitur tersebut? karena keduanya memiliki korelasi yang cukup baik dan mengandung informasi yang sama, yaitu luas distribusi berdasarkan fungsi pairplot().
* Setelah dilakukan proses PCA, terdapat fitur baru bernama 'dimension'. Itu adalah fitur hasil pengurangan dimensi dari fitur 'highway MPG ' dan 'city mpg'.
* Kemduian, terdapat proses mengetahui proporsi dari kedua komponen tersebut yang menghasilkan kumpulan array yang merupakan Principal Component (PC). PC pertama maksudnya informasi mengenai dua fitur diatas sebagian besar terdapat di PC tersebut, sementara sisanya terdapat di PC dua. Pc pertama ini mewakili dua fitur melalui sebuah variabel baru yang bernama "dimension".

### 3. Train-Test-Split
Proses pembagian data menjadi train data atau data latih dan test data atau data uji. Proses ini dibantu oleh library sklearn dengan metode train_test_split. Untuk proporsi pembagian kedua data, yaitu 80:20.
"""

X = df_final.drop(['Harga_Jual'], axis = 1)
y = df_final['Harga_Jual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

print(f"Total # of sample in whole dataset: {len(X)}")
print(f"Total # of sample in train dataset: {len(X_train)}")
print(f"Total # of sample in test dataset: {len(X_test)}")

"""**Notes:**
* Terlihat hasil dari pembagian data diatas, train data sejumlah 4497 dan test data sejumlah 1125.

### 4. Standarisasi
Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Teknik ini digunakan untuk transformasi pada feature numerik. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Untuk menghindari kebocoran informasi pada data uji, penerapan dilakukan untuk fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, standarisasi dilakukan pada data uji.
"""

numerical_features = ['Engine HP',	'Engine Cylinders', 'dimension']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

"""**Notes:**
* Berdasarkan output diatas, terlihat benar bukan bahwa standarisasi mengubah nilai mean menjadi 0 dan nilai standar devisiasi menjadi 1. Sekitar 68% dari nilai akan berada di antara -1 dan 1.
* Berikut pendukung kuat dari hasil proses tersebut.
"""

X_train[numerical_features].describe().round(3)

"""## Model Devlopment
Proses kali ini menunjukan penggunaan machine learning dengan beberapa Algoritma yang akan digunakan. Algoritma yang akan digunakan pada proses Model Devploment kali ini ada tiga, yaitu K-Nearest Neighbor, Random Forest, dan Boosting Algorithm. Sebelum itu, pembuatan DataFrame yang berisi ketiga algoritma diatas untuk membandingkan hasil prediksi terbaik perlu dibuat.
"""

models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""### 1. Model Devlopment Menggunakan Algoritma K-Nearest Neighbor atau KNN
KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.

Kali ini, modelling dengan Algoritma KNN akan menggunakan bantuan hyperparameter tuning untuk menemukan kombinasi nilai optimal untuk hyperparameter dari sebuah model machine learning dengan tujuan meningkatkan performa model.
"""

knn = KNeighborsRegressor()
parameters_knn = {
    'n_neighbors' : [10, 20, 30, 40, 50, 60, 70, 80]
}


random_search_knn = GridSearchCV(knn, parameters_knn, scoring='neg_mean_squared_error', cv=5)

random_search_knn.fit(X_train,y_train)
print(f"RandomizedSearch score for KNN: {random_search_knn.best_score_}")
print("RandomizedSearch params for KNN: ")
print(random_search_knn.best_params_)

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','KNN'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""**Notes:**
* Berdasarkan pencarian parameter tuning dengan bantuan GridSearch dan metrix MSE, output muncul untuk nilai 'n_neighbors': 10 dan untuk skor evaluasi terbaik adalah -23874737.26177538.
* Kemudian, output tersebut akan digunakan dalam pemodelan menggunakan Algoritma KNN.

### 2. Model Devlopment Menggunakan Algoritma Random Forest

Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama.


Kali ini, modelling dengan Algoritma Random Forest akan menggunakan bantuan hyperparameter tuning untuk menemukan kombinasi nilai optimal untuk hyperparameter dari sebuah model machine learning dengan tujuan meningkatkan performa model.
"""

parameters_RandomForest = {
    'n_estimators': [30, 40, 50, 60, 70, 80],
    'max_depth': [16, 32, 64, 128],
}


random_search_RF = GridSearchCV(RandomForestRegressor(random_state=55, n_jobs=-1),parameters_RandomForest, scoring='neg_mean_squared_error', cv=5)

random_search_RF.fit(X_train,y_train)
print(f"RandomizedSearch score for RF: {random_search_RF.best_score_}")
print("RandomizedSearch params for RF: ")
print(random_search_RF.best_params_)

RF = RandomForestRegressor(n_estimators=80, max_depth=64, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""**Notes:**
* Berdasarkan pencarian parameter tuning dengan bantuan GridSearch dan metrix MSE, output muncul nilai dari beberapa parameter seperti max_depth': 64,'n_estimators': 80, serta skor terbaik adalah -14088820.61456519
* Kemudian, output tersebut akan digunakan dalam pemodelan menggunakan Algoritma Random Forest.
* Untuk parameter lain yang digunakan dalam peermodelan tersebut seperti random_state adalah penambahan manual untuk menentukan seed untuk generator angka acak yang digunakan saat membagi data menjadi subset saat membangun pohon sehingga menghasilkan hasil yang sama setiap kali kode dijalankan
* Sementara, untuk n_jobs bertujuan untuk menentukan jumlah pekerjaan yang akan dijalankan paralel saat melatih model.

### 3. Model Devlopment Menggunakan Algoritma Boosting Algorithm
Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.

Kali ini, modelling dengan Boosting Algorithm akan menggunakan bantuan hyperparameter tuning untuk menemukan kombinasi nilai optimal untuk hyperparameter dari sebuah model machine learning dengan tujuan meningkatkan performa model.
"""

parameters_boosting = {
    'learning_rate': [0.1, 0.01, 0.05],
}

random_search_boosting = GridSearchCV(AdaBoostRegressor(random_state=55), parameters_boosting, scoring='neg_mean_squared_error', cv=5)

random_search_boosting.fit(X_train, y_train)
print(f"RandomizedSearch score for Boosting: {random_search_boosting.best_score_}")
print("RandomizedSearch params for Boosting: ")
print(random_search_boosting.best_params_)

boost = AdaBoostRegressor(learning_rate=0.1, random_state=55)
boost.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boost.predict(X_train), y_true=y_train)

"""**Notes:**
* Berdasarkan pencarian parameter tuning dengan bantuan GridSearch dan metrix MSE, output muncul nilai dari beberapa parameter seperti learning_rate': 0.1.
* Kemudian, output tersebut akan digunakan dalam pemodelan menggunakan Algoritma AdaBoost.
* Untuk parameter lain yang digunakan dalam permodelan tersebut seperti random_state yang merupakan penambahan manual untuk menentukan seed untuk generator angka acak yang digunakan saat membagi data menjadi subset saat membangun pohon sehingga menghasilkan hasil yang sama setiap kali kode dijalankan

## Evaluasi Model
Tahap evaluasi dalam membangun model bertujuan untuk mengukur kinerja dan keefektifan model yang telah dibuat. Evaluasi model penting karena memberikan pemahaman tentang seberapa baik model dapat melakukan prediksi atau menggeneralisasi data baru yang tidak terlihat selama pelatihan. Pada tahap ini, penggunaan metrik akan dilakukan. Metrik yang akan digunakan adalah Mean Squared Eror atau MSE. Metrik ini menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi melalui sebuah persamaan.

 Jadi, proses dalam tahap evaluasi adalah scaling numeric features, count MSE for data train and test, prediction target variable, Calculates the difference between the predicted value and the y_true value, and Model Accuracy Based on Each Algorithm.

### 1. Scaling numeric features

Sebelum menghitung nilai MSE, proses scaling akan dilakukan untuk fitur numerik pada data uji. Karena sebelumnya, pada proses scaling hanya pada data latih saja. Setelah model dilatih menggunakan 3 jenis algoritma yaitu KNN, Random Forest dan AdaBoost, proses scaling fitur akan dilakukan pada data uji. Hal ini harus dilakukan agar skala antara data latih dan data uji sama sehingga evaluasi dapat berjalan.
"""

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

"""### 2. Count MSE for data train and test
Saat menghitung nilai Mean Squared Error pada data train dan test, perhitungan nilai tersebut akan membagi kedua data dengan nilai 1e3. Hal ini bertujuan agar nilai mse berada dalam skala yang tidak terlalu besar. Untuk memudahkan dalam pembacaan hasil diatas maka akan plot visualisasi akan diterapkan.
"""

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RandomForest','Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RandomForest': RF, 'Boosting': boost}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

# Panggil mse
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""### 3. Prediction target variable
Hasil prediksi nilai target berdasarkan ketiga model algoritma(KNN, RandomForest,dan Boosting) menggunakan beberapa harga dari data test.
"""

prediksi = X_test.iloc[:10].copy()
pred_dict = {'y_true':y_test[:10]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)

"""### 4. Calculates the difference between the predicted value and the y_true value
Untuk menghitung rata-rata selisih antara nilai aktual (y_true) dan nilai prediksi yang dihasilkan oleh tiga model berbeda: KNN (K-Nearest Neighbors), RandomForest, dan Boosting (AdaBoost).
"""

selisih_KNN = np.abs(pred_dict['y_true'] - pred_dict['prediksi_KNN'])
selisih_RF = np.abs(pred_dict['y_true'] - pred_dict['prediksi_RandomForest'])
selisih_Boosting = np.abs(pred_dict['y_true'] - pred_dict['prediksi_Boosting'])

rata_selisih_KNN = selisih_KNN.mean()
rata_selisih_RF = selisih_RF.mean()
rata_selisih_Boosting = selisih_Boosting.mean()

print(f"Rata-rata selisih KNN: {rata_selisih_KNN}")
print(f"Rata-rata selisih RandomForest: {rata_selisih_RF}")
print(f"Rata-rata selisih Boosting: {rata_selisih_Boosting}")

"""### 5. Model Accuracy Based on Each Algorithm
 Dengan menggunakan metode score() untuk semua model yang diuji dengan metrik yang sama, membandingkan kinerja relatif dari berbagai model dapat dilakukan. Ini membantu dalam pemilihan model terbaik untuk digunakan dalam situasi tertentu.
"""

pred1 = knn.predict(X_test)
score1 = knn.score(X_test, y_test)
print(f'Akurasi model dengan Algoritma KNN adalah {score1}')

pred2 = boost.predict(X_test)
score2 = boost.score(X_test, y_test)
print(f'Akurasi model dengan Algoritma Boosting adalah {score2}')

pred3 = RF.predict(X_test)
score3 = RF.score(X_test, y_test)
print(f'Akurasi model dengan Algoritma RandomForest adalah {score3}')

"""**Notes:**
* Pada tahap evaluasi ini, model Random Forest (RF) menghasilkan skor nilai error paling kecil dibandingkan algoritma lain seperti KNN dan Boosting Algorithm.
* Model dengan algoritma tersebut juga memiliki nilai absolut dari rata-rata selisih antara nilai aktual (y_true) dan nilai prediksi yang terkecil. Itu mengindikasikan bahwa  model cenderung memiliki kinerja yang lebih baik. Dalam konteks prediksi, nilai selisih yang lebih kecil menunjukkan bahwa prediksi model lebih mendekati nilai aktual.
* Terlihat kembali bahwa pada bagian akurasi model menggunakan algoritma Random Forest menghasilkan skor yang cukup baik, yaitu 0.9313177979043304.
* Dapat disimpulkan bahwa Model Random Forest yang akan dipilih sebagai model terbaik untuk memprediksi harga jual mobil.

## Conclusion
Berdasarkan beberapa tahapan yang telah dilakukan, kesimpulan diperoleh sebagai berikut:
* Pemahaman mengenai Domain Proyek yang berisi latar belakang proyek, tujuan dan kebutuhan proyek, dan riset atau sumber referensi yang relevan dengan proyek telah disertakan.
* Pemahaman mengenai kasus bisnis yang berisi problem statemnets, goals, dan solution statements telah terselesaikan dan terbuktikan hasilnya.
* Pendeskripsian dan pemahaman mengenai informasi data-data, sumber dara, visualisasi data dan fitur/variabel yang digunakan sangat mendukung tahap-tahap pra-permodelan.
* Tahapan persiapan data yang terdiri Encoding fitur kategori, reduksi dengan Principal Component Analysis(PCA), Pembagian dataset, dan teknik standarisasi berhasil untuk dilakukan.
* Proses modeling melalui perbandingan tiga algoritma regresi seperti K-Nearest Neighbor, Random Forest, dan Bosting Algorithm menjadi acuan dalam pemilihan model proyek.
* Tahap evaluasi model menunjukan hasil yang berbeda-beda dengan terpilihnya Model Random Forest sebagai model terbaik untuk prediksi harga jual proyek ini.
"""