"""
//------------------------------------------------------------
Title: main.py
Author: Mark Macapagal
Description: Main program to run k-DAE algorithm
Date: June 2024
Version: 1.0
Changes: Initial release
Note: This code requires the src folder containing k_dae.py, autoencoder.py, and
        utils.py files in the same directory.                        
//------------------------------------------------------------

"""
## Libraries and modules
#==============================external libraries==============================#
from src.k_dae import KDae 
from src.utils import utils

#==============================libraries in python==============================#
from scipy.spatial.distance import hamming #
from sklearn import metrics 
from pandas import ExcelWriter 
import numpy as np 
import pandas as pd 
import logging 
from pathlib import Path
import os
import argparse
import timeit
import smtplib 
from email.message import EmailMessage

os.environ["CUDA_VISIBLE_DEVICES"]="0"  

if __name__ == '__main__':
    print("=========================================== Initializing Simulation ... ============================================")
    parser = argparse.ArgumentParser()

    parser.add_argument('-sd', '--save_dir',
                        type=str,
                        default='save',
                        help='path to save')

    parser.add_argument('-dn', '--dataset_name',
                        choices=['cost','imt2020'],
                        default='cost',
                        help='dataset name [cost,imt2020] ')
    
    parser.add_argument('-ds_num', '--dataset_number',
                        type=int,
                        choices= range(1, 9),
                        default=1,
                        help='dataset number [1,2,3,4,5,6,7,8] ')
    
    parser.add_argument('-sn', '--sheet_number',
                        type=int,
                        choices= range(1,31),
                        default=1,
                        help='dataset sheet number [1:30] ')
    
    parser.add_argument('-create', '--create_new_excel',
                        type=int,
                        choices= range(0,2),
                        default=1,
                        help='1 for create a new excel file, 0 for not creating a new excel file [0,1] ')

    ## Parse the arguments
    FLAGS, unparsed = parser.parse_known_args()
    save_dir_name = FLAGS.save_dir
    dataset_name = FLAGS.dataset_name
    dataset_number = FLAGS.dataset_number
    sheet_number = FLAGS.sheet_number
    create_new_excel = FLAGS.create_new_excel

    xlsxname, dataset_name = utils.menu('Sample', dataset_name, dataset_number)
    clustering_results_path = os.path.join('clustering_results', dataset_name)
    full_clustering_results_path = os.path.join(clustering_results_path, f'{xlsxname}.xlsx')
    print("=========================================== Save Directory & Dataset Name ==========================================")
    print(f"Save Directory: {save_dir_name}, Dataset Name: {dataset_name}")
    path_dir = Path(os.path.join(save_dir_name, dataset_name))
    print("=========================================== Creating directory... ==================================================")
    print(f"Creating directory... at: {path_dir}")
    path_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory created or already exists: {path_dir.resolve()}")
    print("=========================================== Configuring k_dae logger... ============================================")
    print("Building k_dae logger...")
    log_path = os.path.join(save_dir_name, dataset_name, 'k_dae.log')
    print(f"k_dae log file built: {log_path}")
    utils.config_logger(log_path)
    print("=========================================== Starting k_dae logger... ===============================================")
    logging.debug('Start running dataset name - {}'.format(dataset_name))
    print("")

    
    utils.excel(dataset_name)
    print("=========================================== Excel file name ========================================================")
    print(f"Excel file created: {xlsxname}.xlsx")
    while True:
        try:
            print()
            print('0 - Proceed')
            print('1 - Create a New Excel File')
            print()
            if create_new_excel == 1:
                df1 = pd.DataFrame()
                df1.to_excel(full_clustering_results_path, index=False)
                print('full_clustering_results_path:', full_clustering_results_path)
                print("Excel file created: {}.xlsx".format(xlsxname))
                break
            elif create_new_excel == 0:
                print('full_clustering_results_path:', full_clustering_results_path)
                print("Excel file already created: {}.xlsx".format(xlsxname))
                break
            else:
                print('Invalid Option. Please try again.')
                print()
        except ValueError:
            print('Invalid Option. Please try again.')
    print()
   
    x_train, y_train, sheet_name = utils.load_data(dataset_name, xlsxname, sheet_number)    
    tic = timeit.default_timer()
    print('================================ Initializing k-DAE model variablies ... ============================================')
    n_cluster = len(np.unique(y_train))
    model = KDae(number_cluster=n_cluster, k_dae_epoch=40, epoch_ae=10, initial_epoch=80, dataset_name=dataset_name)
    print(f'sheet_name: {sheet_name}')
    print("=========================================== Running k-DAE model... ==================================================")
    try:
        print("===================================== Fitting the k-DAE model... ====================================================")
        model.fit(x_train, y_train, dataset_name=dataset_name)
        print("================================ Predicting the k-DAE model... ======================================================")
        y_pred = model.predict(x_train)
        print("================================ Getting reconstruction errors... ===================================================")
        reconstruction_errors = model.get_reconstruction_errors()
        print("Reconstruction Errors:", reconstruction_errors)
        print("=========================================== Clustering Performance... ===============================================")
        k_means_nmi, k_means_acc, k_means_ari, k_means_jac = utils.cluster_performance(y_pred, y_train)
    except Exception as e:
        logging.error(f"An error occurred during model fitting or prediction: {e}")
        print(f"An error occurred: {e}")
        exit(1)   
    
    n_pred = len(np.unique(y_pred))
    toc = timeit.default_timer()
    mon, sec = divmod(toc-tic, 60)
    hr, mon = divmod(mon, 60)    
    
    print()
    print("=========================================== Initial Clustering Only... ==============================================")
    print("***Initial Clustering Only***")
    # Read y_initial from k_dae.log automatically
    log_path = os.path.join(save_dir_name, dataset_name, 'k_dae.log')
    with open(log_path, 'r') as log_file:
        for line in log_file:
            if "Numbers in num_list are:" in line:
                y_initial_raw = line.split("Numbers in num_list are:")[1].strip()
                print(f"Extracted y_initial: {y_initial_raw}")
                break
        else:
            raise ValueError("y_initial not found in k_dae.log")

    y_initial = list(map(int, y_initial_raw.split()))
    y_initial = np.array(y_initial)
    n_initial = len(np.unique(y_initial))
    k_means_nmi_ini, k_means_acc_ini, k_means_ari_ini, k_means_jac_ini = utils.cluster_performance(y_initial, y_train)
    print(f'y_initial: {y_initial}, \ny_train: {y_train} \nn_initial: {n_initial}')

    col_num = int(sheet_name.replace('Sheet', ''))
    print('======================================== Saving results to Excel... =================================================')
    print(f'Column Number to be saved: Sheet{col_num}')
    df1 = pd.DataFrame({'y_train':y_train,'y_initial':y_initial,'y_pred':y_pred})
    df2 = pd.DataFrame(data=None,columns=["Sheet "+str(col_num)])
    df3 = pd.DataFrame(data=None,columns=['','NMI_ini','ARI_ini','ACC_ini','JAC_ini',
                                          'NMI','ARI','ACC','JAC','TRUE','INITIAL',
                                          'PRED','TIME','Reconstruction errors'])
    df4 = pd.DataFrame({'':"Sheet "+str(col_num),'NMI_ini':k_means_nmi_ini,'ARI_ini'
                        :k_means_ari_ini,'ACC_ini':k_means_acc_ini,'JAC_ini':
                        k_means_jac_ini,'NMI':k_means_nmi,'ARI':k_means_ari,'ACC'
                        :k_means_acc,'JAC':k_means_jac,'TRUE':n_cluster,'INITIAL'
                        :n_initial,'PRED':n_pred,'TIME':"%d:%02d:%02d" % (hr, mon, sec),
                        'Reconstruction Error': reconstruction_errors},index=[0])
    writer = ExcelWriter(full_clustering_results_path, mode='a', if_sheet_exists='overlay') 
    df1.to_excel(writer, sheet_name='Sheet1', index=False, startrow=1, 
                 startcol=((col_num - 1)*3))
    df2.to_excel(writer, sheet_name='Sheet1', index=False, startrow=0, 
                 startcol=((col_num - 1)*3))
    df3.to_excel(writer, sheet_name='Sheet2', index=False, startrow=1, startcol=0)
    df4.to_excel(writer, sheet_name='Sheet2', index=False, startrow=(col_num + 1), 
                 startcol=0,header=False)
    writer.close()

    def print_summary():
        print('======================================== Results Summary of Clustering ===============================================')
        print(f'Column Number to be saved: Sheet{col_num}')
        print(f"Dataset Name: {dataset_name}")
        print(f"Number of True Clusters: {n_cluster}")
        print(f"Number of Initial Clusters: {n_initial}")
        print(f"Number of Predicted Clusters: {n_pred}")
        print(f"NMI Initial: {k_means_nmi_ini}, NMI Predicted : {k_means_nmi}")   
        print(f"ARI Initial: {k_means_ari_ini}, ARI Predicted : {k_means_ari}")
        print(f"ACC Initial: {k_means_acc_ini}, ACC Predicted : {k_means_acc}")
        print(f"JAC Initial: {k_means_jac_ini}, JAC Predicted : {k_means_jac}") 
        print(f"Reconstruction Errors: {reconstruction_errors}")    
        print("Elapsed time:", toc, tic)
        print("Elapsed time during the whole program in seconds:","%d:%02d:%02d" % (hr, mon, sec))  
        print("=========================================== Simulation Completed  ====================================================")

    def get_email_summary():
        summary = []
        summary.append("==== Results Summary of Clustering ====")
        summary.append(f'Column Number to be saved: Sheet{col_num}')
        summary.append(f"Dataset Name: {dataset_name}")
        summary.append(f"Number of True Clusters: {n_cluster}")
        summary.append(f"Number of Initial Clusters: {n_initial}")
        summary.append(f"Number of Predicted Clusters: {n_pred}")
        summary.append(f"NMI Initial: {k_means_nmi_ini}, \nNMI Predicted : {k_means_nmi}")   
        summary.append(f"ARI Initial: {k_means_ari_ini}, \nARI Predicted : {k_means_ari}")
        summary.append(f"ACC Initial: {k_means_acc_ini}, \nACC Predicted : {k_means_acc}")
        summary.append(f"JAC Initial: {k_means_jac_ini}, \nJAC Predicted : {k_means_jac}") 
        summary.append(f"Reconstruction Errors: {reconstruction_errors}")    
        summary.append("Elapsed time during the whole program in seconds: %d:%02d:%02d" % (hr, mon, sec))  
        summary.append("==== Simulation Completed! ====")
        return "\n".join(summary)

    sender_email = "faithlixmarkmacapagal@gmail.com"
    sender_password = "ptjz fhdm lqmg jrus"  # Use an app password for Gmail/Outlook
    receiver_email = "faithlixmarkmacapagal@gmail.com"
    subject = f"k-DAE {dataset_name} {xlsxname}.xlsx {sheet_name} Simulation Finished!"
    body = f"Your Python simulation has successfully completed.\n{get_email_summary()}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content(body)

    print("======================================== Sending email notification ===============================================")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:  # Use appropriate SMTP server and port
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
    print_summary()
    utils.simulation_completion()

    

