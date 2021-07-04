import pandas as pd
import numpy as np

class decision_stump:
    def __init__(self, thres, thres_var, mode, data, weight):
        """
        input: 
            (1) threshold on which variable
            (2) the whole dataset
        """
        self.thres = thres
        self.var = data[str(thres_var)]
        self.label = data['Y']
        self.n = len(data)
        self.df = pd.DataFrame()
        self.df['Y'] = self.label
        self.init_weight = weight
        self.mode = mode
    
    
    def group(self):
        """
        make a split on data due to the threshold
        """
        self.tmp = np.zeros(self.n)
        for i in range(self.n):
            if self.mode == 'less':
                self.tmp[i] = 1 if self.var[i] > self.thres else 0
            elif self.mode == 'large':
                self.tmp[i] = 0 if self.var[i] > self.thres else 1
            else:
                print('mode type error, mode should be either less or large')
        return self.tmp
    
    
    def error(self, prediction):
        """
        if the prediction is correct, means prediction == label, error = 0
        else error = 1
        """
        self.tmp = np.zeros(self.n)
        for i in range(self.n):
            self.tmp[i] = 0 if self.label[i] == prediction[i] else 1
        return self.tmp
    
    
    def werror(self, weight, error):
        """
        calc the weighted error
        """
        self.tmp = np.zeros(self.n)
        for i in range(self.n):
            self.tmp[i] = weight[i] * error[i]
        return self.tmp
    
    
    def wupdate(self, weight, error, werror):
        """
        update the weight due to the mis-classification
        """
        self.MisRate = sum(werror) / sum(weight)
        self.stage = np.log((1-self.MisRate) / self.MisRate)
       
        self.tmp = np.zeros(self.n)
        for i in range(self.n):
            self.tmp[i] = weight[i] * np.exp(self.stage * werror[i])
        return self.tmp, self.stage
    
    
    def train(self):
        self.df['prediction'] = self.group().astype('int')
        self.df['error'] = self.error(self.df['prediction']).astype('int')
        self.df['weight'] = self.init_weight
        self.df['werror'] = self.werror(self.df['weight'], self.df['error'])
        self.df['weight'], self.stage = self.wupdate(self.df['weight'], self.df['error'], self.df['werror'])
        
        return self.df, self.stage