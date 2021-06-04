import numpy as np

class AutoRegression:
    def __init__(self, lags):
        self.lags = lags
        self.a = np.zeros(self.lags+1)
        self.y = np.zeros(self.lags)
        self.Error = []
        self.Y_pred = []
        self.yhat = []
        
    # model training
    def training(self, data, alpha, epoch):
        for epo in range(epoch):
            for ite in range(self.lags, len(data)):
                self.y[:] = data[ite-self.lags:ite]
                y_real = data[ite]
                
                # equation
                y = self.a[0]
                for i in range(self.lags):
                    y += self.a[i+1] * self.y[i]
                error = y - y_real
                if epo == epoch-1:
                    self.yhat.append(y)
#                 self.Error.append(error)
#                 print(f'ite: {ite}, error: {error}, a: {self.a}, y: {self.y}')
                
                # SGD
                self.a[0] -= alpha * error
                for i in range(self.lags):
                    self.a[i+1] -= alpha * error * self.y[i]
        return self.yhat
               
    # predicting new data
    def prediction(self, test_data, history):
        for ite in range(len(test_data)):
            self.y = history[ite:self.lags+ite]
            
            # eq
            y_pred = self.a[0]
            for i in range(self.lags):
                y_pred += self.a[i+1] * self.y[i]
                          
            self.Y_pred.append(y_pred)
            history = np.append(history, y_pred)
        return self.Y_pred