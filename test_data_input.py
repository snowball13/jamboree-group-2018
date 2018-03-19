import glob
from dataextraction import *
# from jamboree import *

if __name__ == '__main__':
    # # get all files
    # lossfiles = glob.glob("AIR_data/Loss_Data/Complete/Region_*_DR.csv")
    # lossfiles = sorted(lossfiles)
    # print(lossfiles)
    # data = extract_data(lossfiles)
    # print(data.keys())

    filename = "AIR_data/Loss_Data/Complete/Region_1_DR.csv"
    data = extract_data(filename)
    print(data.keys())

    # filename = "AIR_data/Predictor_Data/GEM_HistoricalFreq.csv"
    # data = extract_data(filename)
    # print(data.keys())

    # print(len(data['AIRSID']))
    # print(data.keys())
    # data_test = get_test_data(data)
    # print(len(data['AIRSID']))
