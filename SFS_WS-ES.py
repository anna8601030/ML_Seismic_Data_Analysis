import numpy as np
import pandas as pd
import csv
import pathlib
import warnings
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
import os
warnings.simplefilter("ignore")

############################################
data_type   = 'mat'
#              mat: MAT file (MATLAB)
#              csv: CSV file
mat_name    = 'x.mat'

# MAT name, for data_type = 'mat' only
csv_name    = 'Fvect_29f_2016_ELD.csv'
# CSV name, for data_type = 'csv' only
Sta         = 'ELD'
# Station name, for data_type = 'csv' only
N_features  = 29
# Number of features in the feature vector
Scoring     = 5
# Scoring by
#             1: Tremor precision              (FPR)
#             2: Regional earthquake precision (FNR)
#             3: Tremor recall                 (TPR)
#             4: Regional earthquake recall    (TNR)
#             5: Overall accuracy, which is    (TPR + TNR) / 2
result_name = 'SFS_WS-ES_All-station_k50_Result.csv'
# Output file name
############################################
# Start time
start_time = time.time()

#header = '# C11 C22 C33 TR_Recal Overal_Accu'
header = '# C11 C12 C21 C22 TR_Recal Overal_Accu'
FILE = open(result_name, 'w', newline='')

with FILE:
    writer = csv.writer(FILE)
    writer.writerow(header.split())

scaler = StandardScaler()
LOO = LeaveOneOut()

# CSV file
if data_type == 'csv':
    df = pd.read_csv(f'./LeaveOne_Station_Out/Fvect_29f_2016_{Sta}.csv', header=0)
    print(f'Data: {csv_name}\nTremor: {df[df.Label_Num == 0].shape[0]}\nRegional earthquake: {df[df.Label_Num == 1].shape[0]}')

    # Drop NaN value
    df = df.dropna()
    # Drop 'Event' which is the filename of data
    df = df.drop(['Event'], axis=1)

    TR_Preci_SFS = []
    RE_Preci_SFS = []
    TR_Recal_SFS = []
    RE_Recal_SFS = []
    Overal_Accu_SFS = []
    print(f'|--------|-------|--------|--------|--------|--------|--------|')
    print(f'|   {"N":>3}  |  {"#":>3}  |  {"Accu":<4}  |  {"TR_r":<4}  |  {"RE_r":<4}  |  {"TR_p":<4}  |  {"RE_p":<4}  |')
    print(f'|--------|-------|--------|--------|--------|--------|--------|')

    result = np.array([])
    f_selected = np.array([], dtype='int')
    f_unselected = np.linspace(1, N_features, num = N_features, dtype='int')

    for i_selected in range(N_features):
        result_tmp = []

        for i_unselected in range(len(f_unselected)):

            print(f'Calculating... ({i_unselected + 1}/{len(f_unselected)})', end = '\r')

            f = np.append(f_selected, f_unselected[i_unselected] - 1)

            X = df.iloc[:, f]

            # Normalization    
            X_costumize_norm = scaler.fit_transform(X)
            Y = df.Label_Num

            # kNN
            kNN = KNeighborsClassifier(
                                    n_neighbors = 5,
                                    n_jobs      = -1,
                                    weights     = 'distance'
                                    )

            # Prediction
            Y_predicted = cross_val_predict(
                                            kNN,
                                            X_costumize_norm,
                                            Y,
                                            cv      = 2, 
                                            n_jobs  = -1, 
                                            verbose = 0
                                            )
            # Confusion matrix
            confusion_mat = np.array([])
            confusion_mat = confusion_matrix(Y, Y_predicted)

            TR_Preci_tmp = confusion_mat[0][0]/(confusion_mat[0][0]+confusion_mat[1][0])
            RE_Preci_tmp = confusion_mat[1][1]/(confusion_mat[0][1]+confusion_mat[1][1])
            TR_Recal_tmp = confusion_mat[0][0]/(confusion_mat[0][1]+confusion_mat[0][0])
            RE_Recal_tmp = confusion_mat[1][1]/(confusion_mat[1][0]+confusion_mat[1][1])
            Overal_Accu_tmp = (TR_Recal_tmp + RE_Recal_tmp) / 2

            result_tmp.append([f_unselected[i_unselected], TR_Preci_tmp, RE_Preci_tmp, TR_Recal_tmp, RE_Recal_tmp, Overal_Accu_tmp])
        
        f_added_in_this_time = f_unselected[np.argmax(result_tmp, axis = 0)[Scoring]]
        if len(f_selected) != 0:
            if result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][Scoring] > np.max(result[:][Scoring]):
                print(f'|{"+"}  {i_selected + 1:>3}  |  {f_added_in_this_time:>3}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]*100:>5.1f}  |')
            elif result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][Scoring] == np.max(result[:][Scoring]):
                print(f'|{"="}  {i_selected + 1:>3}  |  {f_added_in_this_time:>3}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]*100:>5.1f}  |')
            elif result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][Scoring] < np.max(result[:][Scoring]):
                print(f'|{"-"}  {i_selected + 1:>3}  |  {f_added_in_this_time:>3}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]*100:>5.1f}  |')
        else:
            print(f'|   {i_selected + 1:>3}  |  {f_added_in_this_time:>3}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]*100:>5.1f}  |')

        f_selected = np.append(f_selected, result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][0] - 1)
        result = np.append(result, result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]])
        result_write = f'{result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][0]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]}'

        FILE = open(result_name, 'a', newline='')
        with FILE:
            writer = csv.writer(FILE)
            writer.writerow(result_write.split())

        f_unselected = np.delete(f_unselected, np.argmax(result_tmp, axis = 0)[Scoring], axis=0)

    print(f'|--------|-------|--------|--------|--------|--------|--------|')
    print(f'Result saved to ./{result_name}')

    elapsed_time = time.time() - start_time
    print(f'Elapsed time = {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

# MAT file
elif data_type == 'mat':
    from scipy.io import loadmat

    result = np.array([])
    f_selected = np.array([], dtype='int')
    f_unselected = np.linspace(1, N_features, num = N_features, dtype='int')

    mat_name = '/home/anna/ERI_src/201410-201503_WS_ES-tremor_E_All-stations_feature.mat'

    mat = loadmat(mat_name)
    Fvect = mat['C_data'][0]
    ES_TREMOR = Fvect[1]

    mat_name = '/home/anna/ERI_src/WS_Ecomp_All-station_feature.mat'
    mat = loadmat(mat_name)
    Fvect = mat['C_data'][0]
    W_TREMOR = Fvect[0]
    C_TREMOR = Fvect[1]
    E_TREMOR = Fvect[2]

    ##################################################
    row_rand_W_TREMOR = np.arange(W_TREMOR.shape[0])
    np.random.shuffle(row_rand_W_TREMOR)
    W_TREMOR = W_TREMOR[row_rand_W_TREMOR[0:5000]]
    ##################################################
    ##################################################
    row_rand_C_TREMOR = np.arange(C_TREMOR.shape[0])
    np.random.shuffle(row_rand_C_TREMOR)
    C_TREMOR = C_TREMOR[row_rand_C_TREMOR[0:6272]]
    ##################################################
    ##################################################
    row_rand_E_TREMOR = np.arange(E_TREMOR.shape[0])
    np.random.shuffle(row_rand_E_TREMOR)
    E_TREMOR = E_TREMOR[row_rand_E_TREMOR[0:5000]]
    ##################################################

    WS_TREMOR = np.r_[W_TREMOR, C_TREMOR, E_TREMOR]
    WS_TREMOR = WS_TREMOR.real
    ES_TREMOR = ES_TREMOR.real

    print(f'W-Tremor: {len(WS_TREMOR)}\nES-Tremor: {len(ES_TREMOR)}')

    C11 = []
    C12 = []
    C21 = []
    C22 = []
    TR_Recal_SFS = []
    Overal_Accu_SFS = []
    print(f'|--------|-------|--------|--------|--------|--------|')
    print(f'|   {"N":>3}  |  {"#":>3}  |  {"Accu":<4}  |  {"TR_r":<4}  |  {"C11":<4}  |  {"C22":<4} |')
    #print(f'|   {"N":>3}  |  {"#":>3}  |  {"Accu":<4} |')
    print(f'|--------|-------|--------|--------|--------|--------|')

    df = np.abs(np.append(WS_TREMOR, ES_TREMOR, axis=0))
    df = pd.DataFrame(df)
    # Drop NaN value
    df = df.dropna()

    Y = np.append(np.zeros((len(WS_TREMOR),1), dtype='int'), np.ones((len(ES_TREMOR),1), dtype='int'))
    
    for i_selected in range(N_features):
        result_tmp = []

        for i_unselected in range(len(f_unselected)):

            #print(f'f_unselected:{f_unselected}, f_selected: {f_selected}')
            print(f'Calculating... ({i_unselected + 1}/{len(f_unselected)})', end = '\r')

            f = np.append(f_selected, f_unselected[i_unselected] - 1)

            X = df.iloc[:, f]
            
            # Normalization    
            X_costumize_norm = scaler.fit_transform(X)

            # kNN
            kNN = KNeighborsClassifier(
                                    n_neighbors = 5,
                                    n_jobs      = -1,
                                    weights     = 'distance'
                                    )

            # Prediction
            Y_predicted = cross_val_predict(
                                            kNN,
                                            X_costumize_norm,
                                            Y,
                                            cv      = 50, # k (any intrger): k-fold cross-validation
                                                          # LOO: leave-ont-out cross-validation
                                            n_jobs  = -1, 
                                            verbose = 0
                                            )
            # Confusion matrix
            confusion_mat = np.array([])
            confusion_mat = confusion_matrix(Y, Y_predicted)

            C11 = confusion_mat[0][0]
            C12 = confusion_mat[0][1]
            C21 = confusion_mat[1][0]
            C22 = confusion_mat[1][1]
            TR_Recal_tmp = recall_score(Y, Y_predicted, average='weighted') 
            Overal_Accu_tmp = balanced_accuracy_score(Y, Y_predicted)

            #result_tmp.append([f_unselected[i_unselected], C11, C22, C33, TR_Recal_tmp, Overal_Accu_tmp])
            result_tmp.append([f_unselected[i_unselected], C11, C12, C21, C22, TR_Recal_tmp, Overal_Accu_tmp])


        f_added_in_this_time = f_unselected[np.argmax(result_tmp, axis = 0)[Scoring]]
        if len(f_selected) != 0:
            if result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][Scoring] > np.max(result[:][Scoring]):
                print(f'|{"+"}  {i_selected + 1:>3}  |  {f_added_in_this_time:>3}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][6]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]:>5.1f} | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]:>5.1f}  |')
            elif result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][Scoring] == np.max(result[:][Scoring]):
                print(f'|{"="}  {i_selected + 1:>3}  |  {f_added_in_this_time:>3}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][6]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]:>5.1f} | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]:>5.1f}  |')
            elif result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][Scoring] < np.max(result[:][Scoring]):
                print(f'|{"-"}  {i_selected + 1:>3}  |  {f_added_in_this_time:>3}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][6]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]:>5.1f} | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]:>5.1f}  |')
        else:
            print(f'|   {i_selected + 1:>3}  |  {f_added_in_this_time:>3}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][6]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]*100:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]:>5.1f}  | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]:>5.1f} | {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]:>5.1f}  |')


        f_selected = np.append(f_selected, result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][0] - 1)
        result = np.append(result, result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]])
        result_write = f'{result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][0]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][1]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][2]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][3]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][4]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][5]} {result_tmp[np.argmax(result_tmp, axis = 0)[Scoring]][6]}'

        FILE = open(result_name, 'a', newline='')
        with FILE:
            writer = csv.writer(FILE)
            writer.writerow(result_write.split())

        f_unselected = np.delete(f_unselected, np.argmax(result_tmp, axis = 0)[Scoring], axis=0)

    print(f'|--------|-------|--------|--------|--------|--------|--------|')
    print(f'Result saved to ./{result_name}')

    elapsed_time = time.time() - start_time
    print(f'Elapsed time = {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

# Invalid data type
else:
    print(f'Invalid data type: {data_type}. Data type must be either csv or mat.')


################### End ####################
