from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Akurasi(y_test, y_pred):
    akurasi = np.round(accuracy_score(y_test, y_pred),2)
    return akurasi

def Presisi(y_test, y_pred):
    presisi = np.round(precision_score(y_test, y_pred),2)
    return presisi

def Recall(y_test, y_pred):
    recall = np.round(recall_score(y_test, y_pred),2)
    return recall

def F1(y_test, y_pred):
    f1 = np.round(f1_score(y_test, y_pred),2)
    return f1

def CM(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm

def nilaiCM(cm):
    # Inisialisasi variabel array untuk menyimpan TN, FP, FN, dan TP
    TN = []
    FP = []
    FN = []
    TP = []
    T = []
    F = []

    # Menghitung nilai TN, FP, FN, dan TP dari setiap matriks CM
    for i in range(len(cm)):
        tn, fp, fn, tp = cm[i].ravel()
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        TP.append(tp)
        T.append(tn+tp)
        F.append(fn+fp)
    return TN, FP, FN, TP, T, F

def Metrik(y_test, prediksi):
    akurasi, presisi, recall, f1, cm = [], [], [], [], []
    for i in range(prediksi.shape[1]):
        nilai_akurasi = Akurasi(y_test, prediksi.iloc[:,i])
        nilai_presisi = Presisi(y_test, prediksi.iloc[:,i])
        nilai_recall = Recall(y_test, prediksi.iloc[:,i])
        nilai_f1 = F1(y_test, prediksi.iloc[:,i])
        nilai_cm = CM(y_test, prediksi.iloc[:,i])
        akurasi.append(nilai_akurasi)
        presisi.append(nilai_presisi)
        recall.append(nilai_recall)
        f1.append(nilai_f1)
        cm.append(nilai_cm)
    return akurasi, presisi, recall, f1, cm

def maksMetrik(metrik):
    maks_akurasi = metrik['akurasi'].max()
    maks_presisi = metrik['presisi'].max()
    maks_recall = metrik['recall'].max()
    maks_f1 = metrik['f1'].max()
    k_maks_akurasi = metrik[metrik['akurasi'] == maks_akurasi]['k'].values
    k_maks_presisi = metrik[metrik['presisi'] == maks_presisi]['k'].values
    k_maks_recall = metrik[metrik['recall'] == maks_recall]['k'].values
    k_maks_f1 = metrik[metrik['f1'] == maks_f1]['k'].values
    print(
        "Akurasi tertinggi ",maks_akurasi," pada K ",k_maks_akurasi,"\n"
        "Presisi tertinggi ",maks_presisi," pada K ",k_maks_presisi,"\n"
        "Recall tertinggi ",maks_recall," pada K ",k_maks_recall,"\n"
        "Skor F1 tertinggi ",maks_f1," pada K ",k_maks_f1
    )

def visualisaiMetrik(metrik):
    # Plot grup bar chart
    plt.figure(figsize=(8, 6))
    bar_width = 0.15
    index = np.arange(len(metrik['k']))
    opacity = 0.8

    plt.bar(index, metrik['akurasi'], bar_width, alpha=opacity, label='Akurasi')
    plt.bar(index + bar_width, metrik['presisi'], bar_width, alpha=opacity, label='Presisi')
    plt.bar(index + (2 * bar_width), metrik['recall'], bar_width, alpha=opacity, label='Recall')
    plt.bar(index + (3 * bar_width), metrik['f1'], bar_width, alpha=opacity, label='F1')

    # Menambahkan label pada sumbu x dan y
    plt.xlabel('k')
    plt.ylabel('Nilai')

    # Menambahkan judul
    plt.title('Grafik Hasil Metrik Evaluasi KNN')

    # Mengatur label sumbu x
    plt.xticks(index + bar_width, metrik['k'])

    # Menampilkan legenda di luar plot dan mengatur posisi legenda
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Menampilkan plot
    plt.show()

def visualisasiCMklasifikasi(cm, acak_K):
    # Mengatur ukuran dan layout
    fig, axes = plt.subplots(nrows=1, ncols=len(cm), figsize=(15, 4))

    # Memvisualisasikan setiap matriks CM
    for i, (matrix, k) in enumerate(zip(cm, acak_K)):
        ax = axes[i]
        sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', cbar=False, ax=ax,
            xticklabels=['0', '1'],   
            yticklabels=['0', '1'])
        ax.set_title(f"Confusion Matrix k {k}")
        ax.set_xlabel ('prediksi')       
        ax.set_ylabel ('aktual')    

    # Menampilkan plot
    plt.tight_layout()
    plt.show()
    print()

def visualisasiCM(cm):
    # Membuat heatmap confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)  # Atur skala font
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['0', '1'], 
                yticklabels=['0', '1'])

    # Menambahkan label pada sumbu
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix')

    # Menampilkan plot
    plt.show()