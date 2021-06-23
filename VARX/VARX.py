import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

class VARX:
    def __init__(self, n=2, m=1, k=0):
        """
        n: lags of data
        m: lags of exog data
        k: time shift of exog data
        default: n=2, m=1, k=0, the same as [zhao et. al @IOTDA'20 BigData'20]
        """
        self.n = n
        self.m = m
        self.k = k
        self.a = np.zeros(self.n+1)
        self.b = np.zeros(self.m+1)
        
        self.y_hist = np.zeros(self.n)
        self.u_hist = np.zeros(self.m+1)
        
        self.Error = []
        self.y_pred = []
        self.yhat = []
        self.RMSE = []
    
    def plot_rmse(self):
        plt.plot(self.RMSE)
        plt.xlabel('epoch')
        plt.ylabel('RMSE(loss)')
    
    def training(self, data, data_exog, lr, epoch):
        """
        model training
        input y-data, u-data_exog, learning rate-lr, epoch from outside
        """

        for epo in range(epoch):
            # to record rmse each epoch
            self.yhat = []
            
            for ite in range(max(self.n, self.k+self.m), len(data)):
                self.y_hist[:] = data[ite-self.n : ite]
                self.u_hist[:] = data_exog[ite-self.k-self.m : ite-self.k+1]
                y_real = data[ite] 
                
                # equation
                y = self.a[0]
                for i in range(self.n):
                    y += self.a[i+1] * self.y_hist[i]
                for i in range(self.m+1):
                    y += self.b[i] * self.u_hist[i]
                error = y - y_real
                
                # only record the predicted y in last epoch iteration
                self.yhat.append(y)

                
                # SGD
                self.a[0] -= lr * error
                for i in range(self.n):
                    self.a[i+1] -= lr * error * self.y_hist[i]
                for i in range(self.m+1):
                    self.b[i] -= lr * error * self.u_hist[i]
                
            # RMSE
            rmse = sqrt(mean_squared_error(data[max(self.n, self.k+self.m):], self.yhat))
            self.RMSE.append(rmse)
            print(f'epoch {epo}, RMSE={round(rmse,3)}')


        # to calculate the training RMSE
        # return the predicted y
        return self.yhat
               

    def prediction(self, data, data_exog):
        """
        predict unknown data points
        but the first few data points cannot be used due to lags
        """
        history = data[:max(self.n, self.k+self.m)]
        
        for ite in range(max(self.n, self.k+self.m), len(data)):
            self.y_hist[:] = history[ite-self.n : ite]
            self.u_hist[:] = data_exog[ite-self.k-self.m : ite-self.k+1]
            
            # eq
            y = self.a[0]
            for i in range(self.n):
                y += self.a[i+1] * self.y_hist[i]
            for i in range(self.m+1):
                y += self.b[i] * self.u_hist[i]
                          
            self.y_pred.append(y)
            history = np.append(history, y)
        return self.y_pred