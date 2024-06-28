from sklearn.model_selection import KFold
import pandas as pd

def Split(X_train, y_train):
    # Menginisialisasi objek KFold
    kfold = KFold(n_splits=2, shuffle=False, random_state=None)
    # Lakukan iterasi K-Fold
    for train_index, val_index in kfold.split(X_train):
        X_train_fold1, X_train_fold2 = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold1, y_train_fold2 = y_train.iloc[train_index], y_train.iloc[val_index]
    return X_train_fold1, X_train_fold2, y_train_fold1, y_train_fold2

def SplitFold(dataset, n_split):
    # Inisialisasi KFold dengan 9 fold
    kfold = KFold(n_splits=n_split, shuffle=False)
    # Variabel untuk menyimpan data setiap fold
    fold = []
    # Lakukan iterasi KFold dan simpan data di setiap fold
    for train_index, test_index in kfold.split(dataset):
        fold_data = dataset.iloc[test_index]
        fold.append(fold_data)
    return fold

def PrintFold(fold):
    for i, fold_data in enumerate(fold):
        print(f"Data di fold {i}:")
        print(fold_data)
        print()

def DataTrainig(fold, dataset):
    # Mendapatkan indeks baris yang terdapat dalam folds[0]
    idx_fold = fold.index
    # Menampilkan data dari dataset yang tidak termasuk dalam folds[0]
    data_training = dataset.drop(idx_fold)
    return data_training
