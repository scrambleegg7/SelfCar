#
# Training program designed by H. Hamano
# Feb 26 2018
#

import tensorflow as tf   
import keras
import numpy as np  
import pandas as pd 

import os


def loadData():

    baseDir = "./sim_data2"
    driving_log_file = "driving_log.csv"
    driving_log = os.path.join(baseDir,driving_log_file)

    df_drive = pd.read_csv(driving_log)
    print(df_drive.head() )

    pass


def main():

    loadData()



if __name__ == "__main__":

    main()