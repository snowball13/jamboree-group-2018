import glob
from dataextraction import *
# from jamboree import *
from AnalyseData import *

if __name__ == '__main__':
    # get all files
    lossfiles = glob.glob("AIR_data/Loss_Data/Complete/Region_*_DR.csv")
    lossfiles = sorted(lossfiles)
    print(lossfiles)
    data_loss = extract_data(lossfiles[0:1])
    print(data_loss.keys())

    ids, avg = get_average_lob1(data_loss)

    ids_pred, data_pred  = get_prediction_data()

    avg = get_correct_data_arrays(ids, avg, ids_pred, data_pred)
    print len(ids_pred), len(avg), len(data_pred)

    # print(len(data['AIRSID']))
    # print(data.keys())
    # data_test = get_test_data(data)
    # print(len(data['AIRSID']))
