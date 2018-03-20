import glob
from dataextraction import *
# from jamboree import *

if __name__ == '__main__':
    # get all files
    lossfiles = glob.glob("AIR_data/Loss_Data/Complete/Region_*_DR.csv")
    lossfiles = sorted(lossfiles)
    print(lossfiles)

    data_loss = extract_data(lossfiles)
    print(data_loss.keys())

    region_ids = get_ids_list(data_loss)
    print(len(region_ids))

    # filename = "AIR_data/Loss_Data/Complete/Region_1_DR.csv"
    # data_lob1 = extract_data([filename])
    # print(data_lob1.keys())
    #
    # filename = "AIR_data/Predictor_Data/GEM_HistoricalFreq.csv"
    # data_pred = extract_data([filename])
    # print(data_pred.keys())

    # ids, avg = get_average_lob1(data_loss)

    predfiles = glob.glob("AIR_data/Predictor_Data/*.csv")
    predfiles = sorted(predfiles)
    print(predfiles)

    data_pred = extract_data(predfiles)
    print(data_pred.keys())

    pred_ids = get_ids_list(data_loss)
    print(len(pred_ids))

    # ids, avg, pred = get_correct_data_arrays(ids, avg, data_pred)
    # print len(ids), len(avg), len(pred)

    # print(len(data['AIRSID']))
    # print(data.keys())
    # data_test = get_test_data(data)
    # print(len(data['AIRSID']))
