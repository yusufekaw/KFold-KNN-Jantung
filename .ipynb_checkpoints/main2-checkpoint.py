from data.pemrosesanData import ambilData, infoKolom, labelEncode, MinMax, mappingFitur, cetak_kolom_kategorikal
from algoritma.KNN import K, splitDataset, Prediksi, Jarak, jarakTerdekat, hasilJarakTerdekat, acakK, trainLabel, Klasifikasi, semuaPrediksi, Feature
from algoritma.KFold import Split, SplitFold, PrintFold, DataTrainig
from pengujian.metrik_evaluasi import Akurasi, Presisi, Recall, F1, CM, nilaiCM, Metrik, visualisaiMetrik, maksMetrik, visualisasiCMklasifikasi, visualisasiCM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold

# Main program
if __name__ == '__main__':
    
    # Load dataset
    dataset = ambilData() 
    
    # menampilkan dataset
    print ("\t\tHasil Import Dataset") 
    print (dataset) 

    #informasi kolom dataset
    print("\t\tInformasi Kolom dataset")
    dataset.info() 

    # menampilkan informasi kolom kategorikal (object)
    print("\t\tInformasi Kolom kategorikal")
    cetak_kolom_kategorikal(dataset) 

    # encoding nilai kategorikal menjadi numerikal
    dataset = labelEncode(dataset) 
    
    # menampilkan dataset yang telah diencoding
    print ("\t\tDataset Yang Telah Diencoding")
    print (dataset)

    #nama file untuk menyimpan laporan
    nama_file = "data/dataset/laporan.xlsx"

    # Normalisasi dataset
    dataset = MinMax(dataset) 
    # menampilkan dataset yang telah dinormalisasi
    print ("\t\tDataset Yang Telah dinormalisasi")
    print (dataset)

    #variabel fold
    n_split = 9
    fold = []
    #split datafold
    fold = SplitFold(dataset, n_split)
    
    #inisialisasi 6 K berbeda
    k = [3,5,7,9,11,13]
    maks_K = np.max(k)

    avg_akurasi = avg_presisi = avg_recall = avg_f1 = 0
    metrik = 0

    k_total = 0

    #algoritma KNN
    for i in range(len(fold)):
        #data training
        print(f"Data training ke-{i}:")
        data_training = DataTrainig(fold[i], dataset)
        print(data_training)

        #data testing
        print(f"Data testing ke-{i}:")
        data_testing = fold[i]
        print(data_testing)

        #Split fitur dataset
        X_train, X_test, y_train, y_test = Feature(data_training, data_testing)
        
        #mencari jarak terdekat
        terdekat, indeks = jarakTerdekat(X_test, X_train, y_train, maks_K)
        # Melakukan klasifikasi menggunakan kNN untuk setiap nilai k
        y_pred = semuaPrediksi(k, X_train, y_train, X_test)
        # mencari jarak terdekat
        hasil_jarak_terdekat = hasilJarakTerdekat(X_test, y_train, terdekat, indeks)    
        #konversi data frame
        hasil_jarak_terdekat = pd.DataFrame(hasil_jarak_terdekat, columns=["Testing", "Training", "Jarak", "Kelas"])
        print(hasil_jarak_terdekat)

        #klasifikasi berdasarkan jarak terdekat
        kolom_prediksi = ["K" + str(val) for val in k] # Membuat dictionary kolom
        prediksi = pd.DataFrame(np.transpose(y_pred), columns=kolom_prediksi) # Membuat dataframe
        klasifikasi = Klasifikasi(X_test, prediksi, y_test) # Hasil klasifikasi
        print(klasifikasi)

        akurasi, presisi, recall, f1, cm = Metrik(y_test, prediksi)
        #visualisasiCMklasifikasi(cm, k)

        TN, FP, FN, TP, T, F = nilaiCM(cm)

        nilai_cm = pd.DataFrame({'k': k, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'T': T, 'F': F})

        print(f"Nilai klasifikasi fold {i} berdasarkan matriks konfusi")
        print(nilai_cm)

        metrik_per_fold = pd.DataFrame({'k': k,'akurasi': akurasi, 'presisi': presisi, 'recall': recall, 'f1': f1})
        print(f"metrik evaluasi data fold {i}")
        print(metrik_per_fold)

        #visualisaiMetrik(metrik_per_fold)
        
        if(i==0):
            metrik=metrik_per_fold
        else:
            metrik=pd.concat([metrik, metrik_per_fold], ignore_index=True)
        
    avg_akurasi = round(metrik.groupby('k')['akurasi'].mean(),2)
    avg_akurasi = avg_akurasi.reset_index(drop=True)
    avg_presisi = round(metrik.groupby('k')['presisi'].mean(),2)
    avg_presisi = avg_presisi.reset_index(drop=True)
    avg_recall = round(metrik.groupby('k')['recall'].mean(),2)
    avg_recall = avg_recall.reset_index(drop=True)
    avg_f1 = round(metrik.groupby('k')['f1'].mean(),2)
    avg_f1 = avg_f1.reset_index(drop=True)
    k=pd.DataFrame({'k':k})
    avg_metrik =  pd.concat([k, avg_akurasi, avg_presisi, avg_recall, avg_f1], axis=1)
    print("rata-rata nilai metrik evaluasi")
    print(avg_metrik)
    visualisaiMetrik(avg_metrik)

    '''
    #data training selain dari fold1
    data_training = DataTrainig(fold[0], dataset)
    #print data training
    print("data training:")
    print(data_training)

    

    X_train, X_test, y_train, y_test = Feature(data_training, fold[0]) 

    #mencari ketetanggan terdekat
    terdekat, indeks = jarakTerdekat(X_test, X_train, y_train, maks_K)
    # Melakukan klasifikasi menggunakan kNN untuk setiap nilai k
    y_pred = semuaPrediksi(k, X_train, y_train, X_test)
    # mencari jarak terdekat
    hasil_jarak_terdekat = hasilJarakTerdekat(X_test, y_train, terdekat, indeks)    
    #konversi data frame
    df_hasil_jarak_terdekat = pd.DataFrame(hasil_jarak_terdekat, columns=["Testing", "Training", "Jarak", "Kelas"])
    print(df_hasil_jarak_terdekat)

    kolom_prediksi = ["K" + str(val) for val in k] # Membuat dictionary kolom
    prediksi = pd.DataFrame(np.transpose(y_pred), columns=kolom_prediksi) # Membuat dataframe
    klasifikasi = Klasifikasi(X_test, prediksi, y_test) # Hasil klasifikasi
    print(klasifikasi)
    '''
