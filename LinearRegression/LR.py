class LinearRegression:
    def __init__(self):
        self.b0 = 0.0
        self.b1 = 0.0
        self.B0 = []
        self.B1 = []
        self.Error = []
        self.Y_pred = []
    
    def append_record(self, er, b0, b1):
        self.Error.append(er)
        self.B0.append(b0)
        self.B1.append(b1)
        
    def training(self, train_data, alpha, epoch):
        for i in range(epoch):
            for _, row in train_data.iterrows():
                x = row['x']
                y_real = row['y']
                
                y = self.b0 + self.b1 * x
                error = y - y_real
                self.b0 = self.b0 - alpha * error
                self.b1 = self.b1 - alpha * error * x
                
                self.append_record(error, self.b0, self.b1)
#                 print(self.b0)
                
    def prediction(self, train_data):
        for _, row in train_data.iterrows():
            x = row['x']
            y_pred = self.B0[-1] + self.B1[-1] * x
            self.Y_pred.append(y_pred)
        return self.Y_pred