import csv, random


def extract_data(filename):
     length = 100000
     airsid = zeros(length)
     lob1 = zeros(length)
     # First row of file is used to create the keywords for the subsequent rows that
     # are read in as a dictionary
     i = 0
     with open('example.csv') as csvfile:
          reader = csv.DictReader(csvfile)
          for row in reader:
              airsid[i] = row['AIRSID']
              lob1[i] = row['LOB1']
              i++
     # We only want 20% of the data
     indices = random.sample(range(1, length), int(round(0.2*length)))
     return airsid[indices], lob1[indices]

filename = "AIR_data/Loss_data/Complete/Region_1_DR.csv"
airsid, lobs1 = extract_data(filename)
