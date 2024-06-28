import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#fungsi untuk memuat dataset
def ambilData():
    # Path ke file dataset
    #path = "dataset/heart.csv"
    # Membaca dataset menggunakan Pandas
    dataset = pd.read_csv("data/dataset/heart.csv")
    return dataset

#Informasi kolom bertipe data object
def infoKolom(dataset):
    #mengambil hanya data bertipe object
    kolom_objek = dataset.select_dtypes(include='object').columns
    hasil = {} #variabel untuk menyimpan hasil
    for kolom in kolom_objek:
        nilai_unik = dataset[kolom].nunique() #nilai unik dari kolom
        nilai_perhitungan = dataset[kolom].value_counts() #manghitung jumlah nilai dalam kolom
        nilai_info = {nilai: hitung for nilai, hitung in nilai_perhitungan.items()}
        hasil[kolom] = {
            'nilai_unik': nilai_unik, #nilai dalam kolom
            'nilai_perhitungan': nilai_info #jumlah nilai unik dalam kolom
        }
    return hasil

#Label Encoding mengubah nilai kategorikal menjadi numerikal
def labelEncode(dataset):
    le = LabelEncoder()
    for kolom in dataset.columns:
        if dataset[kolom].dtype == 'object': #mengambil kolom bertipe data objek
            dataset[kolom] = le.fit_transform(dataset[kolom]) #mengganti nilai kategori menjadi angka
    return dataset

#Min-Max Normalisasi
def MinMax(dataset):
    scaler = MinMaxScaler()
    data_normalisasi = dataset.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    data_normalisasi = np.round(data_normalisasi,2)
    data_normalisasi['HeartDisease'] = data_normalisasi['HeartDisease'].astype(int)
    return data_normalisasi

#Fitur Mapping : Mengubah nama fitur
def mappingFitur(dataset):
    nama_fitur = {} #menyimpan nama fitur
    for i, kolom in enumerate(dataset.columns):
        if kolom != 'HeartDisease': #mengubah nama fitur kecuali target
            nama_fitur[kolom] = f'X{i+1}' #mengubah nama menjadi X1,X2 dst
        else:
            nama_fitur[kolom] = 'Y' #target menjadi Y
    dataset.rename(columns=nama_fitur, inplace=True)
    return dataset, nama_fitur

def cetak_kolom_kategorikal(dataset):
    info_kolom = infoKolom(dataset)
    for kolom, info in info_kolom.items():
        nilai_unik = info['nilai_unik']
        nilai_perhitungan = info['nilai_perhitungan']
        print("Kolom :", kolom)
        for nilai, hitung in nilai_perhitungan.items():
            print("\t\t",nilai,"\t: ",hitung," data")
        print("\t",nilai_unik," Nilai")