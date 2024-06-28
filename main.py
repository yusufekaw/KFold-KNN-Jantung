from data.pemrosesanData import ambilData, infoKolom, labelEncode, MinMax, mappingFitur, cetak_kolom_kategorikal
from algoritma.KNN import K, splitDataset, Prediksi, Jarak, jarakTerdekat, hasilJarakTerdekat, acakK, trainLabel, Klasifikasi, semuaPrediksi
from algoritma.KFold import Split
from pengujian.metrik_evaluasi import Akurasi, Presisi, Recall, F1, CM, nilaiCM, Metrik, visualisaiMetrik, maksMetrik, visualisasiCMklasifikasi, visualisasiCM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

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
    #eksport hasil encoding ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       dataset.to_excel(writer, sheet_name="label encoding", index=True)'''

    # Normalisasi dataset
    dataset = MinMax(dataset) 
    # menampilkan dataset yang telah dinormalisasi
    print ("\t\tDataset Yang Telah dinormalisasi")
    print (dataset)

    #eksport hasil normalisasi ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       dataset.to_excel(writer, sheet_name="normalisasi", index=True)'''

    #membuang kolom HeartDisease , digunakan sebagai target
    kolom_target = 'HeartDisease' 
    #split data training dan testing
    X_train, X_test, y_train, y_test = splitDataset(dataset, kolom_target, test_size=0.2, random_state=42)
    
    print("y_train")
    print(y_train)
    print("y_test")
    print(y_test)
    #eksport data training ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       X_train.to_excel(writer, sheet_name="data training", index=True)'''

    #eksport data testing ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       X_test.to_excel(writer, sheet_name="data testing", index=True)'''

    #split kFOld
    X_train_fold1, X_train_fold2, y_train_fold1, y_train_fold2 = Split(X_train, y_train)

    #data training Fold 1
    print("\t\t Data Training KFold 1")
    print(X_train_fold1)

    #eksport data training fold 1 ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       X_train_fold1.to_excel(writer, sheet_name="Train Fold 1", index=True)'''

    #data training Fold 2
    print("\t\t Data Training KFold 2")
    print(X_train_fold2)

    #eksport data training fold 1 ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       X_train_fold2.to_excel(writer, sheet_name="Train Fold 2", index=True)'''

    #menentukan k untuk k-fold
    k_fold = [3] 
    
    #menjadikan data training fold 1 sebagai validasi dan data training fold 2 sebagai testing
    #mencari ketetanggaan terdekat data training 2 ke data training 1
    terdekat_fold2, indeks_fold2 = jarakTerdekat(X_train_fold2, X_train_fold1, y_train_fold1, 3)
    # mencari jarak terdekat
    fold2_hasil_jarak_terdekat = hasilJarakTerdekat(X_train_fold2, y_train_fold1, terdekat_fold2, indeks_fold2) 
    #konversi data frame
    fold2_hasil_jarak_terdekat = pd.DataFrame(fold2_hasil_jarak_terdekat, columns=["Fold2", "Fold1", "Jarak", "Kelas"])
    print(fold2_hasil_jarak_terdekat) #menampilkan hasil jarak terdekat
    
    #eksport data jarak fold 2 ke fold 1 ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       fold2_hasil_jarak_terdekat.to_excel(writer, sheet_name="Jarak Fold 2 ke 1", index=True)'''

    
    #validasi kelas data training fold2
    y_validasi_fold2 = semuaPrediksi(k_fold, X_train_fold1, y_train_fold1, X_train_fold2)
    #konversi kelas validasi menjadi data frame
    y_validasi_fold2 = pd.DataFrame(np.transpose(y_validasi_fold2), columns=['validasi'])
    #ambil index data training fold2
    y_validasi_fold2.index = X_train_fold2.index
    #cetak kelas hasil validasi
    y_train_subset = y_train.loc[y_validasi_fold2.index]
    combined_df_fold2 = pd.concat([y_train_subset.rename('target'), y_validasi_fold2], axis=1)
    print(combined_df_fold2)

    #eksport hasil validasi fold 2 ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       combined_df_fold2.to_excel(writer, sheet_name="validasi fold 2", index=True)'''

    #menjadikan data training fold 2 sebagai validasi dan data training fold 1 sebagai testing
    #mencari ketetanggaan terdekat data training 2 ke data training 1
    terdekat_fold1, indeks_fold1 = jarakTerdekat(X_train_fold1, X_train_fold2, y_train_fold2, 3)
    # mencari jarak terdekat
    fold1_hasil_jarak_terdekat = hasilJarakTerdekat(X_train_fold1, y_train_fold2, terdekat_fold1, indeks_fold1) 
    #konversi data frame
    fold1_hasil_jarak_terdekat = pd.DataFrame(fold1_hasil_jarak_terdekat, columns=["Fold1", "Fold2", "Jarak", "Kelas"])
    print(fold1_hasil_jarak_terdekat) #menampilkan hasil jarak terdekat

    #eksport data jarak fold 1 ke fold 2 ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       fold1_hasil_jarak_terdekat.to_excel(writer, sheet_name="Jarak Fold 1 ke 2", index=True)'''

    #validasi kelas data training fold2
    y_validasi_fold1 = semuaPrediksi(k_fold, X_train_fold2, y_train_fold2, X_train_fold1)
    #konversi kelas validasi menjadi data frame
    y_validasi_fold1 = pd.DataFrame(np.transpose(y_validasi_fold1), columns=['validasi'])
    #ambil index data training fold2
    y_validasi_fold1.index = X_train_fold1.index
    #cetak kelas hasil validasi
    y_train_subset = y_train.loc[y_validasi_fold1.index]
    combined_df_fold1 = pd.concat([y_train_subset.rename('target'), y_validasi_fold1], axis=1)
    print(combined_df_fold1)

    #eksport hasil validasi fold 1 ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       combined_df_fold1.to_excel(writer, sheet_name="validasi fold 1", index=True)'''

    hasil_validasi = pd.concat([combined_df_fold1,combined_df_fold2])
    print(hasil_validasi)

    #eksport hasil validasi data training ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       combined_df_fold1.to_excel(writer, sheet_name="validasi training", index=True)'''

    for index, value in hasil_validasi['validasi'].items():
      y_train.loc[index] = value
    print(y_train)

    cm_validasi =  CM(hasil_validasi['target'], hasil_validasi['validasi'])
    visualisasiCM(cm_validasi)
    TN_validasi, FP_validasi, FN_validasi, TP_validasi = cm_validasi.ravel()
    print(
            'TN : ',TN_validasi,'\n'
            'FP : ',FP_validasi,'\n'
            'FN : ',FN_validasi,'\n'
            'TP : ',TP_validasi,'\n'
          )
    akurasi_validasi = Akurasi(hasil_validasi['target'], hasil_validasi['validasi'])
    print("Akurasi validasi : ", akurasi_validasi)
    presisi_validasi = Presisi(hasil_validasi['target'], hasil_validasi['validasi'])
    print("Presisi validasi : ", presisi_validasi)
    recall_validasi = Recall(hasil_validasi['target'], hasil_validasi['validasi'])
    print("Recall validasi : ", recall_validasi)
    f1_validasi = F1(hasil_validasi['target'], hasil_validasi['validasi'])
    print("F1 Score validasi : ", f1_validasi)
    
    acak_K = acakK()
    maks_K = np.max(acak_K)
    print("nilai k yang dihasilkan :", acak_K)

    #mencari ketetanggan terdekat
    terdekat, indeks = jarakTerdekat(X_test, X_train, y_train, maks_K)
    # Melakukan klasifikasi menggunakan kNN untuk setiap nilai k
    y_pred = semuaPrediksi(acak_K, X_train, y_train, X_test)
    # mencari jarak terdekat
    hasil_jarak_terdekat = hasilJarakTerdekat(X_test, y_train, terdekat, indeks)    
    #konversi data frame
    df_hasil_jarak_terdekat = pd.DataFrame(hasil_jarak_terdekat, columns=["Testing", "Training", "Jarak", "Kelas"])
    print(df_hasil_jarak_terdekat)

    #eksport hasil perhitungan jarak terdekat ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       df_hasil_jarak_terdekat.to_excel(writer, sheet_name="jarak training testing", index=True)'''

    kolom_prediksi = ["K" + str(val) for val in acak_K] # Membuat dictionary kolom
    prediksi = pd.DataFrame(np.transpose(y_pred), columns=kolom_prediksi) # Membuat dataframe
    klasifikasi = Klasifikasi(X_test, prediksi, y_test) # Hasil klasifikasi
    print(klasifikasi)

    #eksport hasil klasifikasi ke excel
    '''with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       klasifikasi.to_excel(writer, sheet_name="klasifikasi", index=True)'''

    akurasi, presisi, recall, f1, cm = Metrik(y_test, prediksi)
    visualisasiCMklasifikasi(cm, acak_K)

    TN, FP, FN, TP, T, F = nilaiCM(cm)

    nilai_cm = pd.DataFrame({'k': acak_K, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'T': T, 'F': F})

    print("Nilai prediksi berdasarkan matriks konfusi")
    print(nilai_cm)

    #eksport confusion matrix ke excel
    with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       nilai_cm.to_excel(writer, sheet_name="confusion matrix", index=True)
    
    print("Mengurutkan K berdasarkan nilai True terbaik")
    print(nilai_cm.sort_values(by='T', ascending=False))

    metrik = pd.DataFrame({'k': acak_K,'akurasi': akurasi, 'presisi': presisi, 'recall': recall, 'f1': f1})
    print("metrik evaluasi")
    print(metrik)

    #eksport metrik evaluasi ke excel
    with pd.ExcelWriter(nama_file, engine="openpyxl", mode="a") as writer:
       metrik.to_excel(writer, sheet_name="metrik", index=True)

    #maksMetrik(metrik)
    urut_akurasi = metrik[['k','akurasi']].sort_values(by='akurasi', ascending=False)
    urut_presisi = metrik[['k','presisi']].sort_values(by='presisi', ascending=False)
    urut_recall = metrik[['k','recall']].sort_values(by='recall', ascending=False)
    urut_f1 = metrik[['k','f1']].sort_values(by='f1', ascending=False)
    print("\takurasi terbaik")
    print(urut_akurasi.T)
    print("\tpresisi terbaik")
    print(urut_presisi.T)
    print("\trecall terbaik")
    print(urut_recall.T)
    print("\tf1 terbaik")
    print(urut_f1.T)

    visualisaiMetrik(metrik)