import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import euclidean_distances, pairwise_distances


# Menentukan rentang nilai K
min_k = 3
max_k = 10

# Memilih nilai K secara acak
def K():
	return random.randint(min_k, max_k)

# 5 nilai k secara acak
def acakK():
    random.seed(69) # Mengatur biji acak 
    k = random.sample(range(min_k, max_k), 5) # Menghasilkan angka acak antara 3 hingga 10
    return k

# split dataset training dan testing
def splitDataset(dataset, kolom_target, test_size=0.2, random_state=42):
    X = dataset.drop(kolom_target, axis=1)
    y = dataset[kolom_target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def Prediksi(X_train, y_train, X_test, k):
    # Inisialisasi model KNN
    knn_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    
    # Melatih model dengan data training
    knn_model.fit(X_train, y_train)
    
    # Melakukan prediksi pada data testing
    y_pred = knn_model.predict(X_test)  
      
    # Mengembalikan hasil prediksi dan jarak
    return y_pred


#def Jarak(X_test, X_train, y_train):
    '''knn_model = KNeighborsClassifier(734)
    knn_model.fit(X_train, y_train)
    distances, indices = knn_model.kneighbors(X_test, return_distance=True)
    jarak = []
    for i in range(len(X_test)):
        baris = []
        for j in range(len(indices[i])):
            nilai = distances[i][j]
            indeks = (i,indices[i][j])
            baris.append((indeks, nilai)) 
            #print("Jarak:", distances[i][j], "Index:", indices[i][j])
        #print()
        jarak.append(baris) 
    jarak = pd.DataFrame(jarak)
    #jarak.index = [X_test.index[i] for i in range(len(X_test))]
    return jarak'''
    
def Jarak(X_test, X_train):
     # Menghitung jarak antara data testing dan data training
    jarak = euclidean_distances(X_test, X_train)
    jarak = np.round(jarak, 2)
    jarak = pd.DataFrame(jarak, columns=[X_train.index[i] for i in range(len(X_train))])
    jarak.index = [X_test.index[i] for i in range(len(X_test))]
    return jarak


def jarakTerdekat(X_test, X_train, y_train, k):
    # Mendapatkan jarak dan indeks tetangga terdekat
    knn_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # Melatih model dengan data training
    knn_model.fit(X_train, y_train)

    distances, indices = knn_model.kneighbors(X_test)
    terdekat = np.round(distances,2) #jarak terdekat
    indeks = indices #indeks
    return terdekat, indeks

def hasilJarakTerdekat(X_test, y_train, terdekat, indeks):
    hasil_jarak_terdekat = []
    #klasifikasi = []
    for i in range(len(X_test)):
        #data_klasifikasi = [X_test.index[i], y_pred[i], y_test.values[i]] #kombinasi index data testing dengan hasil prediksi
        #klasifikasi.append(data_klasifikasi) #disimpan dalam variabel klasifikasi
        for j in range(0, len(indeks[i])):
            #print(X_test.index[i], "\t", terdekat[i][j], "\t", indeks[i][j] ,"\t", y_train.iloc[indeks[i][j]])
        #print()
            #mendapatkan nilai indeks, jarak dan kelas
            data_jarak_terdekat = [X_test.index[i], indeks[i][j], terdekat[i][j], y_train.iloc[indeks[i][j]]]
            #menggabungkan nilai diatas
            hasil_jarak_terdekat.append(data_jarak_terdekat)
    return hasil_jarak_terdekat#, klasifikasi

def trainLabel(X_fold, fold_prediksi):
    indexing = []
    for i in range(len(X_fold)):
        idx = [X_fold.index[i]] #kombinasi index data testing dengan hasil prediksi
        indexing.append(idx) #disimpan dalam variabel klasifikasi
    df_idx = pd.DataFrame(indexing)
    label = df_idx.join(fold_prediksi)
    return label
    
def Klasifikasi(X_test, prediksi, y_test):
    klasifikasi = []
    for i in range(len(X_test)):
        data_klasifikasi = [X_test.index[i], y_test.values[i]] #kombinasi index data testing dengan hasil prediksi
        klasifikasi.append(data_klasifikasi) #disimpan dalam variabel klasifikasi
    df_klasifikasi = pd.DataFrame(klasifikasi, columns=["#", "target"])
    hasil_klasifikasi = df_klasifikasi.join(prediksi)
    return hasil_klasifikasi

def semuaPrediksi(acak_K, X_train, y_train, X_test):
    y_pred = []
    for k in acak_K:
        prediksi = Prediksi(X_train, y_train, X_test, k)
        y_pred.append(prediksi)
    return y_pred
'''
def hasilJarakTerdekat(X_test, y_pred, y_test, y_train, terdekat, indeks):
    hasil_jarak_terdekat = []
    #klasifikasi = []
    for i in range(len(X_test)):
        #data_klasifikasi = [X_test.index[i], y_pred[i], y_test.values[i]] #kombinasi index data testing dengan hasil prediksi
        #klasifikasi.append(data_klasifikasi) #disimpan dalam variabel klasifikasi
        for j in range(0, len(indeks[i])):
            #print(X_test.index[i], "\t", terdekat[i][j], "\t", indeks[i][j] ,"\t", y_train.iloc[indeks[i][j]])
        #print()
            #mendapatkan nilai indeks, jarak dan kelas
            data_jarak_terdekat = [X_test.index[i], indeks[i][j], terdekat[i][j], y_train.iloc[indeks[i][j]]]
            #menggabungkan nilai diatas
            hasil_jarak_terdekat.append(data_jarak_terdekat)
    return hasil_jarak_terdekat#, klasifikasi
    
'''

def Feature(data_training, data_testing):
    X_train = data_training.drop(columns=['HeartDisease'])
    X_test = data_testing.drop(columns=['HeartDisease'])
    y_train = data_training['HeartDisease']
    y_test = data_testing['HeartDisease']
    return X_train, X_test, y_train, y_test