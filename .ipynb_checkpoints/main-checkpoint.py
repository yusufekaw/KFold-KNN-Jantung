from data.pemrosesanData import ambilData, infoKolom, labelEncode, MinMax, mappingFitur, cetak_kolom_kategorikal
from algoritma.KNN import K, splitDataset, Prediksi, Jarak, jarakTerdekat, hasilJarakTerdekat, acakK, Klasifikasi, semuaPrediksi
from pengujian.metrik_evaluasi import Akurasi, Presisi, Recall, F1, CM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Main program
if __name__ == '__main__':
    
    dataset = ambilData() # Load dataset
    
    print ("\t\tHasil Import Dataset") 
    print (dataset) # menampilkan dataset

    
    print("\t\tInformasi Kolom dataset")
    dataset.info() #informasi kolom dataset

    
    print("\t\tInformasi Kolom kategorikal")
    cetak_kolom_kategorikal(dataset) # menampilkan informasi kolom kategorikal (object)

    
    dataset = labelEncode(dataset) # encoding nilai kategorikal menjadi numerikal
    
    print ("\t\tDataset Yang Telah Diencoding")
    print (dataset) # menampilkan dataset yang telah diencoding

    dataset = MinMax(dataset) # Normalisasi dataset
    print ("\t\tDataset Yang Telah dinormalisasi")
    print (dataset) # menampilkan dataset yang telah dinormalisasi
    
    '''
    # Mengubah nama fitur
    dataset, nama_fitur = mappingFitur(dataset)
    
    # menampilkan perubahan nama fitur
    print ("\t\tMenampilkan Perubahan Nama Fitur")
    for lama, baru in nama_fitur.items():
        print(lama, ":", baru)

    # menampilkan dataset yang telah diubah namafiturnya
    print ("\t\tMenampilkan Dataset Yang Telah Diubah Nama Fiturnya")
    print (dataset)
    '''

    acak_K = acakK()
    maks_K = np.max(acak_K)
    print("nilai k acak yang dihasilkan :", acak_K)

    kolom_target = 'HeartDisease' #membuang kolom HeartDisease , digunakan sebagai target
    k = K()
    print("Nilai K yang dihasilkan : ",k)
    #split dataset (training dan testing)
    X_train, X_test, y_train, y_test = splitDataset(dataset, kolom_target, test_size=0.2, random_state=42)

    print("Data Training\n") 
    print(X_train) # menampilkan data training
    print("Data Testing\n") 
    print(X_test) # menampilkan data testing

    '''
    #menampilkan jarak data training dengan data testing
    jarak = Jarak(X_test, X_train)
    print(jarak.transpose())
    '''    
    
    #mencari ketetanggan terdekat
    terdekat, indeks = jarakTerdekat(X_test, X_train, y_train, maks_K)

    #prediksi kelas
    #y_pred = Prediksi(X_train, y_train, X_test, k)
    # Melakukan klasifikasi menggunakan kNN untuk setiap nilai k
    y_pred = semuaPrediksi(acak_K, X_train, y_train, X_test)
    

    # mencari jarak terdekat
    hasil_jarak_terdekat = hasilJarakTerdekat(X_test, y_train, terdekat, indeks)    
    
    #konversi data frame
    df_hasil_jarak_terdekat = pd.DataFrame(hasil_jarak_terdekat, columns=["Testing", "Training", "Jarak", "Kelas"])
    print(df_hasil_jarak_terdekat)

    # Membuat dictionary kolom
    kolom_prediksi = ["K" + str(val) for val in acak_K]
    print(kolom_prediksi)
    # Membuat dataframe
    prediksi = pd.DataFrame(np.transpose(y_pred), columns=kolom_prediksi)
    print(prediksi)

    klasifikasi = Klasifikasi(X_test, prediksi, y_test)
    print(klasifikasi)

    print(klasifikasi.iloc[:,0])
    print(klasifikasi.shape[1])
        
    akurasi = []
    presisi = []
    recall = []
    f1 = []
    cm = []
    for i in range(prediksi.shape[1]):
        nilai_akurasi = Akurasi(y_test, prediksi.iloc[:,i])
        nilai_presisi = Presisi(y_test, prediksi.iloc[:,i])
        nilai_recall = Recall(y_test, prediksi.iloc[:,i])
        nilai_f1 = F1(y_test, prediksi.iloc[:,i])
        nilai_cm = CM(y_test, prediksi.iloc[:,i])
        akurasi.append(nilai_akurasi)
        presisi.append(nilai_akurasi)
        recall.append(nilai_akurasi)
        f1.append(nilai_akurasi)
        cm.append(nilai_cm)
    
    metrik = pd.DataFrame({'k': acak_K,'akurasi': akurasi, 'presisi': presisi, 'recall': recall, 'f1': f1})
    print(metrik)
    print(cm)

    '''
    df_klasifikasi = pd.DataFrame(klasifikasi, columns=["Testing", "Kelas", "Target"])
    print(df_klasifikasi)
    
    #metrik evaluasi
    akurasi = Akurasi(y_test, y_pred)
    presisi = Presisi(y_test, y_pred)
    recall = Recall(y_test, y_pred)
    f1 = F1(y_test, y_pred)
    cm = CM(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    print("Metrik Evaluasi")
    print("Akurasi\t\t: ", akurasi)
    print("Presisi\t\t: ", presisi)
    print("Recall\t\t: ", recall)
    print("F1-score\t: ", f1)
    print("Matriks Konfusi\t:\n", cm)
''' 
'''
   #menjalankan algortima KNN
    y_pred, jarak, terdekat, indeks = knn(X_train, y_train, X_test, k)

    hasil_jarak_terdekat = [] #u/ menyimpan data hasil perhitungan jarak terdekat
    klasifikasi = [] #u/ menyimpan data klasifikasi
    for i in range(len(X_test)):
        baris = [X_test.index[i], y_pred[i], y_test.values[i]] #kombinasi index data testing dengan hasil prediksi
        klasifikasi.append(baris) #disimpan dalam variabel klasifikasi
        for j in range(k):
            ii = indeks[i][j] #indeks
            #mendapatkan nilai indeks, jarak dan kelas
            baris = [X_test.index[i], ii , np.round(terdekat[i][j],2), dataset.Y[ii]]
            #menggabungkan nilai diatas
            hasil_jarak_terdekat.append(baris)

    #konversi data frame
    df_hasil_jarak_terdekat = pd.DataFrame(hasil_jarak_terdekat, columns=["Testing", "Training", "Jarak", "Kelas"])
    print(df_hasil_jarak_terdekat)

    df_klasifikasi = pd.DataFrame(klasifikasi, columns=["Testing", "Kelas", "Target"])
    print(df_klasifikasi)

    # Menghitung dan mencetak metrik evaluasi
    print("Metrik Evaluasi")
    print(f"Akurasi\t\t: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Presisi\t\t: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall\t\t: {recall_score(y_test, y_pred):.2f}")
    print(f"F1-score\t: {f1_score(y_test, y_pred):.2f}")
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    print("Matriks Konfusi:\n", cm)
    # Mencetak perbandingan prediksi benar dan prediksi salah
    print("Jumlah prediksi benar (True)\t:", TN + TP)
    print("Jumlah prediksi salah (False)\t:", FP + FN)
'''