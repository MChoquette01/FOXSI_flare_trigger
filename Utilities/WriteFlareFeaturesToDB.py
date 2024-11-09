from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import numpy as np
import netCDF4 as nc
import math
from pymongo import MongoClient
import emission_measure as emt
import importlib
importlib.reload(emt)

""" Using the PyMongo Database made in WriteFlareFitsToDB.py, additional features are added to each database entry.
"""

class AddTempEMFeatures:
    
    def __init__(self):
        ''' Opens MongoDB Flares database
        '''
        # Open MongoDB
        myclient = MongoClient("mongodb://localhost:27017/")
        flares_db = myclient["Flares"]
        self.flares_table = flares_db["Flares"]
        
    def get_xrs_lists(self):
        '''Returns the XRS and n-minute difference XRS values as lists for temperature and emission measure calculations'''
        #Current XRSA and XRSB:
        self.xrsa_list = []
        self.xrsb_list = []
        #1 minute difference XRSA and XRSB
        self.xrsa_1min_list = []
        self.xrsb_1min_list = []
        #3 minute difference XRSA and XRSB
        self.xrsa_3min_list = []
        self.xrsb_3min_list = []
        #5 minute difference XRSA and XRSB
        self.xrsa_5min_list = []
        self.xrsb_5min_list = []
        #flare id list:
        self.flare_id_list = []
        #making the list structures:
        entries = self.flares_table.find()
        for i in entries:
            self.xrsa_list.append(i['CurrentXRSA'])
            self.xrsb_list.append(i['CurrentXRSB'])
            self.xrsa_1min_list.append(i['XRSA1MinuteDifference'])
            self.xrsb_1min_list.append(i['XRSB1MinuteDifference'])
            self.xrsa_3min_list.append(i['XRSA3MinuteDifference'])
            self.xrsb_3min_list.append(i['XRSB3MinuteDifference'])
            self.xrsa_5min_list.append(i['XRSA5MinuteDifference'])
            self.xrsb_5min_list.append(i['XRSB5MinuteDifference'])
            self.flare_id_list.append(i['FlareID'])
        
    def calculate_temp_em(self):
        ''' Creates lists for the temperature and emission measures calculated from XRS values and differences.''' 
        self.em, self.temp = emt.compute_goes_emission_measure(self.xrsa_list, self.xrsb_list, 16)

        # self.em_1min, self.temp_1min = self.do_temp_em_diff_calculation(self.xrsa_1min_list, self.xrsb_1min_list)
        # self.em_3min, self.temp_3min = self.do_temp_em_diff_calculation(self.xrsa_3min_list, self.xrsb_3min_list)
        # self.em_5min, self.temp_5min = self.do_temp_em_diff_calculation(self.xrsa_5min_list, self.xrsb_5min_list)
        
    def do_delta_temp_em_calculations(self):
        """ Does the straightforward 1-5 minute differences of the temperature and emission measure
        """
    
        
    def do_temp_em_diff_calculation(self, xrsa_list, xrsb_list):
        ''' Calculates temperature and emission measure based off the n-minute XRS differences. 
        NOTE: the calculations come out as nan when there has been a decrease in either XRSA or XRSB (this is because
        we take the ratio of the two, and it doens't make sense if either/both are negative)
        '''
        xrsa_arr = np.array(xrsa_list)
        xrsb_arr = np.array(xrsb_list)
        #need to take out all the None's before calculating temp and em
        none_arr = np.where(xrsb_arr==None)[0]
        no_none_xrsa = np.delete(xrsa_arr, none_arr) 
        no_none_xrsb = np.delete(xrsb_arr, none_arr)
        #also need to take out the few XRSB zeros
        zeros = np.where(no_none_xrsb==0)[0] 
        final_xrsa = np.delete(no_none_xrsa, zeros)
        final_xrsb = np.delete(no_none_xrsb, zeros)
        #actually calculating the tmep and em
        em, temp = emt.compute_goes_emission_measure(final_xrsa, final_xrsb, 16)
        #adding zeros back in:
        temp = np.insert(np.array(temp), zeros, None)
        em = np.insert(np.array(em), zeros, None)
        #adding back the None values:
        temp = np.insert(temp, none_arr, None)
        em = np.insert(em, none_arr, None)
        return em, temp
        
        
    def save_new_features(self):
        '''Saves the temperature and emission measure features into the database.'''
        # for i in range(10):
        #     self.flares_table.update_one({'FlareID':self.flare_id_list[i]}, {'$set': {"Temperature":self.temp[0],
        #                     "EmissionMeasure":self.em[i],
        #                     "Temperature1MinuteDifference":self.temp_1min[i],
        #                     "EmissionMeasure1MinuteDifference":self.em_1min[i],
        #                     "Temperature3MinuteDifference":self.temp_3min[i],
        #                     "EmissionMeasure3MinuteDifference":self.em_3min[i],
        #                     "Temperature5MinuteDifference":self.temp_5min[i],
        #                     "EmissionMeasure5MinuteDifference":self.em_5min[i]}})
        #     print('done')
        for _ in tqdm(self.flare_id_list, desc='Adding Entries...'):
            for i, flareid in enumerate(self.flare_id_list):
                self.flares_table.update_one({'FlareID':flareid}, {"$set": {"Temperature":self.temp[i],
                        "EmissionMeasure":self.em[i],
                        "Temperature1MinuteDifference":self.temp_1min[i],
                        "EmissionMeasure1MinuteDifference":self.em_1min[i],
                        "Temperature3MinuteDifference":self.temp_3min[i],
                        "EmissionMeasure3MinuteDifference":self.em_3min[i],
                        "Temperature5MinuteDifference":self.temp_5min[i],
                        "EmissionMeasure5MinuteDifference":self.em_5min[i]}})
                        
        
        
if __name__ == '__main__':
    test = AddTempEMFeatures()
    test.get_xrs_lists()
    test.calculate_temp_em()
    test.save_new_features()