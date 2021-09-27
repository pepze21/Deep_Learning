# import warnings
# warnings.filterwarnings('ignore')

import numpy as np

## Function

def gradient(machine, param):
    
    if param.ndim == 1:
        temp_param = param
        delta = 0.00005
        learned_param = np.zeros(param.shape)

        for index in range(len(param)):
            target_param = float(temp_param[index])
            temp_param[index] = target_param + delta
            param_plus_delta = machine(temp_param)
            temp_param[index] = target_param - delta
            param_minus_delta = machine(temp_param)
            learned_param[index] = (param_plus_delta - param_minus_delta) / (2 * delta)
            temp_param[index] = target_param

        return learned_param

    elif param.ndim == 2:
        temp_param = param
        delta = 0.00005
        learned_param = np.zeros(param.shape)
    
        rows = param.shape[0]
        columns = param.shape[1]
    
        for row in range(rows):
            for column in range(columns):
                target_param = float(temp_param[row, column])
                temp_param[row, column] = target_param + delta            
                param_plus_delta = machine(temp_param)
                temp_param[row, column] = target_param - delta            
                param_minus_delta = machine(temp_param)
                learned_param[row, column] = (param_plus_delta - param_minus_delta) / (2 * delta)
                temp_param[row, column] = target_param
        
        return learned_param

def sigmoid(x):
    y_hat = 1 / (1 + np.exp(-x))
    return y_hat

## Class

class LogicGate:

    def __init__(self, gate_Type, X_input, y_output):
        self.Type = gate_Type # gate_Type 문자열 지정 Member
        self.X_input = X_input.reshape(4, 2) # X_input Member 초기화
        self.y_output = y_output.reshape(4, 1) # y_output Member 초기화
        self.W = np.random.rand(2, 1) # W Member 초기화
        self.b = np.random.rand(1) # b Member 초기화
        self.learning_rate = 0.01 # learning_rate Member 지정
        
    # Cost_Function(CEE) Method
    def cost_func(self):
        z = np.dot(self.X_input, self.W) + self.b
        y_hat = sigmoid(z)
        delta = 0.00001
        return -np.sum(self.y_output * np.log(y_hat + delta) + (1 - self.y_output) * np.log((1 - y_hat) + delta))
    
    #Learning Method
    def learn(self):
        machine = lambda x : self.cost_func() # type of machine = ??, function?
        print('Initial Cost = ', self.cost_func())
        
        for step in  range(10001):
            self.W = self.W - self.learning_rate * gradient(machine, self.W)
            self.b = self.b - self.learning_rate * gradient(machine, self.b)
    
            if (step % 1000 == 0):
                print('Step = ', step, ', Cost = ', self.cost_func())

    # Predict Method
    def predict(self, input_data):        
        z = np.dot(input_data, self.W) + self.b
        y_prob = sigmoid(z)
    
        if y_prob > 0.5:
            result = 1
        else:
            result = 0
    
        return y_prob, result


X_input  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_output = np.array([0, 0, 0, 1])

AND_Gate = LogicGate('AND_GATE', X_input, y_output)
AND_Gate.learn()

print(AND_Gate.Type, '\n')
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for input_data in test_data:
    (sigmoid_val, logical_val) = AND_Gate.predict(input_data) 
    print(input_data, ' = ', logical_val)







