import glob
from dataextraction import *
# from jamboree import *

if __name__ == '__main__':
    # get all files
    lossfiles = glob.glob("AIR_data/Loss_Data/Complete/Region_*_DR.csv")
    lossfiles = sorted(lossfiles)
    print(lossfiles)
    data_lob1 = extract_data(lossfiles[0:1])
    print(data_lob1.keys())

    # filename = "AIR_data/Loss_Data/Complete/Region_1_DR.csv"
    # data_lob1 = extract_data([filename])
    # print(data_lob1.keys())

    filename = "AIR_data/Predictor_Data/GEM_HistoricalFreq.csv"
    data_pred = extract_data([filename])
    print(data_pred.keys())

    ids, avg = get_average_lob1(data_lob1)
    ids, avg, pred = get_correct_data_arrays(ids, avg, data_pred)
    print len(ids), len(avg), len(pred)

    # print(len(data['AIRSID']))
    # print(data.keys())
    # data_test = get_test_data(data)
    # print(len(data['AIRSID']))
