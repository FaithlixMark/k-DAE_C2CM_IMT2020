"""
//------------------------------------------------------------
 Title: utils.py
 Description: Utility functions for k-DAE algorithm
 Date: June 2024
 Version: 1.0
 Changes: Initial release
 Note: This code requires the src folder containing k_dae.py and autoencoder.py files in the same directory.
//------------------------------------------------------------
"""
## Libraries and modules
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import logging
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial.distance import hamming
#import excel file
import pandas as pd
#Removing Initial Cluster
import os
import winsound

class utils:
    def relabel(labels):
        keys = list(dict.fromkeys(labels))
        return [keys.index(label) for label in labels]
    #logs the data to the k_dae.log notepad
    def config_logger(log_path='k_dae.log'):
        logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    def menu(ExcelName):
        """ Menu for selecting datasets and filenames"""
        while True:
            print("=========================================== Selecting datasets... ==================================================")
            print('Choose dataset:\nC2CM Datasets: cost\nIMT-2020 Datasets: imt2020')
            data = str(input('Select datasets: '))
            print()
            match data:
                case 'cost':
                    print("=========================================== C2CM dataset is selected ===============================================")
                    cost_options = [
                        '01_Indoor_B1_LOS_Single_Results',
                        '02_Indoor_B2_LOS_Single_Results',
                        '03_SemiUrban_B1_LOS_Single_Results',
                        '04_SemiUrban_B2_LOS_Single_Results',
                        '05_SemiUrban_B1_NLOS_Single_Results',
                        '06_SemiUrban_B2_NLOS_Single_Results',
                        '07_SemiUrban_B1_LOS_Multiple_Results',
                        '08_SemiUrban_B2_LOS_Multiple_Results'
                    ]
                    for i, name in enumerate(cost_options, 1):
                        print(f"{i} - {name}")
                    print("=========================================== C2CM dataset is selected ===============================================")
                    while True:
                        try: 
                            option = int(input('Select a number for the filename: '))
                            if 1 <= option <= len(cost_options):
                                ExcelName = cost_options[option - 1]
                                print(ExcelName + '\n')
                                return ExcelName, data
                            else:
                                print('Invalid Option.')
                        except ValueError:
                            print('Invalid Option.')
                case 'imt2020':
                    print("========================================= IMT-2020 dataset is selected =============================================")
                    imt_2020_options = [
                        '01_InH_A_LOS_Results',
                        '02_InH_A_NLOS_Results',
                        '03_RMa_A_LOS_Results',
                        '04_RMa_A_NLOS_Results',
                        '05_UMa_A_LOS_Results',
                        '06_UMa_A_NLOS_Results',
                        '07_UMi_A_LOS_Results',
                        '08_UMi_A_NLOS_Results'
                    ]
                    for i, name in enumerate(imt_2020_options, 1):
                        print(f"{i} - {name}")
                    print("========================================= IMT-2020 dataset is selected =============================================")
                    while True:
                        try:
                            option = int(input('Select a number for the filename: '))
                            if 1 <= option <= len(imt_2020_options):
                                ExcelName = imt_2020_options[option - 1]
                                print(ExcelName + '\n')
                                return ExcelName, data
                            else:
                                print('Invalid Option.')
                        except ValueError:
                            print('Invalid Option.')
                case _:
                    print('Invalid dataset option. Please try again.')
            
    def excel(data):
            """ Select excel datasets and remove initial cluster if exists """
            print("======================================= Selecting file_path of datasets... =========================================")
            print( f'Dataset Selected: {data}')
            if data == 'cost':
                print('C2CM dataset is selected')
                file_path = r'C:\Users\Mark Macapagal\Desktop\MasterCode2\save\cost\initial_cluster.npy'
                print(f"file_path: {file_path}")
            elif data == 'imt2020':
                print('IMT-2020 dataset is selected')
                file_path = r'C:\Users\Mark Macapagal\Desktop\MasterCode2\save\imt2020\initial_cluster.npy'
                print(f"file_path: {file_path}")
            print()
            print('=========================================== Removing Initial Cluster if exists... ==================================')
            if os.path.exists(file_path):
                os.remove(file_path)
                print('Initial Cluster has been Deleted')
            else:
                print('No Initial Cluster Found.')

    def load_data(data_name,xlsxname):
        print("=========================================== Loading datasets... ====================================================")
        print( f'Dataset Selected: {data_name}')
        cost_files = {
            '01_Indoor_B1_LOS_Single_Results': '01_Indo_B1_LOS_Sing_white_new_labels.xlsx',
            '02_Indoor_B2_LOS_Single_Results': '02_Indo_B2_LOS_Sing_white_new_labels.xlsx',
            '03_SemiUrban_B1_LOS_Single_Results': '03_SemiUr_B1_LOS_Sing_white_newlabels.xlsx',
            '04_SemiUrban_B2_LOS_Single_Results': '04_SemiUr_B2_LOS_Sing_white_newlabels.xlsx',
            '05_SemiUrban_B1_NLOS_Single_Results': '05_SemiUr_B1_NLOS_Sing_white_newlabels.xlsx',
            '06_SemiUrban_B2_NLOS_Single_Results': '06_SemiUr_B2_NLOS_Sing_white_newlabels.xlsx',
            '07_SemiUrban_B1_LOS_Multiple_Results': '07_SemiUr_B1_LOS_Mul_white_newlabels.xlsx',
            '08_SemiUrban_B2_LOS_Multiple_Results': '08_SemiUr_B2_LOS_Mul_white_newlabels.xlsx'
        }
        imt_2020_files = {
            '01_InH_A_LOS_Results': '01_InH_A_LOS_DCT.xlsx',
            '02_InH_A_NLOS_Results': '02_InH_A_NLOS_DCT.xlsx',
            '03_RMa_A_LOS_Results': '03_RMa_A_LOS_DCT.xlsx',
            '04_RMa_A_NLOS_Results': '04_RMa_A_NLOS_DCT.xlsx',
            '05_UMa_A_LOS_Results': '05_UMa_A_LOS_DCT.xlsx',
            '06_UMa_A_NLOS_Results': '06_UMa_A_NLOS_DCT.xlsx',
            '07_UMi_A_LOS_Results': '07_UMi_A_LOS_DCT.xlsx',
            '08_UMi_A_NLOS_Results': '08_UMi_A_NLOS_DCT.xlsx'
        }
        base_path = r'C:\Users\Mark Macapagal\Documents\k-DAE C2CM and IMT-2020 Mastercode\k_DAE_C2CM_and_IMT_2020_Mastercode\DATASETS'
        file_dict = cost_files if data_name == 'cost' else imt_2020_files if data_name == 'imt2020' else None
        sub_folder = 'C2CM_DATASET' if data_name == 'cost' else 'IMT2020' if data_name == 'imt2020' else None
        if file_dict and xlsxname in file_dict:
            filename = file_dict[xlsxname]
            filepath = f"{base_path}\{sub_folder}\{filename}"
            print(f"Processing filepath for: {xlsxname}")
            print(f"filepath: {filepath}")
            while True:
                try:
                    sheet_Num = input('Enter a valid Sheet Number (1-30): ')
                    sheet_Num_int = int(sheet_Num)
                    sheet_name = 'Sheet' + str(sheet_Num_int)
                    if 1 <= sheet_Num_int <= 30:
                        print()
                        t = pd.read_excel(filepath, sheet_name=sheet_name)
                        print(t)
                        data = t.to_numpy()
                        x_train = data[:, :-2]
                        y_train = utils.relabel(data[:, -1])
                        print('============================================ Data Loaded Successfully ===============================================')
                        print(f"x_train shape: {x_train.shape}, y_train shape: {np.array(y_train).shape}, sheet_name: {sheet_name}")
                        return x_train, y_train, sheet_name
                    else:
                        print('Invalid Sheet Number. Please try again.')
                except ValueError:
                    print('Invalid input. Please enter a numeric Sheet Number.')
        else:
            print('NO FILE SELECTED. Please try again.')
            return None     
        
    def acc(y_true, y_pred):

        """ Calculate clustering accuracy

        Require scikit-learn installed

        :param y_true: true labels
        :param y_pred: predicted labels
        :return: accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        
        ind = linear_assignment(w.max() - w)
        ind=np .array(list(zip(*ind)))
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


    def k_means(x_train, n_class, n_init=100):
        """ compute k_means algorithm

        use scikit-learn to compute k-means

        :param x_train: data points
        :param n_class: number of clusters
        :param n_init: The number of different initialization
        :return: k_means model
        """
        k_mean = KMeans(n_clusters=n_class, n_init=n_init)
        km_model = k_mean.fit(x_train)
        return km_model

    def cluster_performance(y_pred, y_train, label='kmean'):
        """ calculate performance of clustering


        :param y_pred: Predication vector
        :param y_train: Ground truth vector
        :param label: Method name
        :return: NMI, ACC, ARI
        """    
        k_means_nmi = metrics.normalized_mutual_info_score(y_train, y_pred)
        k_means_ari = metrics.adjusted_rand_score(y_train, y_pred)
        k_means_acc = utils.acc(np.int0(y_train), y_pred)
        k_means_jac = 1 - hamming(np.int0(y_train),utils.relabel(y_pred))
        print('{} NMI is {}'.format(label, k_means_nmi))
        print('{} ARI is {}'.format(label, k_means_ari))
        print('{} Acc is {}'.format(label, k_means_acc))
        print('{} Jac is {}'.format(label, k_means_jac))
        logging.info("NMI - {:0.9f},ARI - {:0.2f},ACC - {:0.2f},JAC - {:0.2f}".format(k_means_nmi, k_means_ari, k_means_acc,k_means_jac))
        logging.info("Numbers in num_list are: {}".format(' '.join(map(str, y_pred))))
        return k_means_nmi, k_means_acc, k_means_ari, k_means_jac

    def simulation_completion():
        """ Signal completion of clustering process """
        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)
    

     
            

       
        
            
      
        
    
      
       
       