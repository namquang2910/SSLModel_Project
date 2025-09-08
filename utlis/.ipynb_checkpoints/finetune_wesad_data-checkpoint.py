import os
import pickle
from utlis.wesad_processing import load_process_extract_ls


def process_wesad_dataset(root_dir, sample_rate, window, stride):
    folder_ls = os.listdir(root_dir)
    folder_ls = [f for f in folder_ls if f not in [".ipynb_checkpoints", ".DS_Store"]]

    out_dir = f'WESAD_{sample_rate}_{window}_{stride}'
    os.makedirs(out_dir, exist_ok=True)

    for file in folder_ls:
        X, y = load_process_extract_ls(root_dir, [file], sample_rate, window, stride, False)

        path = os.path.join(out_dir, file + '.pkl')
        with open(path, "wb") as f:
            pickle.dump((X, y), f)   # Save both X and y as a tuple

#process_wesad_dataset('/home/van/NamQuang/Dataset/WESAD_LOSO', sample_rate = 700, window = 10, stride = 10)

def load_pre_wesad_dataset(root_dir, test_file):
    folder_ls = os.listdir(root_dir)
    folder_ls = [f for f in folder_ls if f not in [".ipynb_checkpoints", ".DS_Store"]]
    
    test_ls = [test_file]
    train_ls = [subject for subject in folder_ls if subject not in test_ls]
    X_train, X_test, y_train, y_test = [], [], [], []
    for f in train_ls:
        path = os.path.join(root_dir, f)
        with open(path, "rb") as file:
            X, y = pickle.load(file)
            X_train.extend(X)
            y_train.extend(y)
            
    for f in test_ls:
        path = os.path.join(root_dir, f)
        with open(path, "rb") as file:
            X, y = pickle.load(file)
            X_test.extend(X)
            y_test.extend(y)
    return X_train, X_test, y_train, y_test



def load_wesad_dataset(root_dir, test_subject):
    folder_ls = os.listdir(root_dir)
    for i in folder_ls:
        if i == ".ipynb_checkpoints" or i == '.DS_Store':
            print(f"Removing {i}")
            folder_ls.remove(i)
    
    train_ls = [subject for subject in folder_ls if subject not in valid_ls]

    # Create the train list by excluding test_ls
    train_ls = [subject for subject in folder_ls if subject not in valid_ls]
    print("==========Loading Training set============")
    X_train, y_train = load_process_extract_ls(root_dir, train_ls,700, 10,10,False)
    print("==========Loading Testing set============")
    X_test, y_test = load_process_extract_ls(root_dir,valid_ls,700, 10, 10,False)
    return X_train, X_test, y_train, y_test