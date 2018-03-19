from dataextraction import *
from jamboree import *

if __name__ == '__main__':

    filename = "AIR_data/Loss_data/Complete/Region_1_DR.csv"
    data = extract_data(filename)
    print(len(data['AIRSID']))
    print(data.keys())
    data_test = get_test_data(data)
    print(len(data['AIRSID']))