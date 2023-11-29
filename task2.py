import sys

import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QListWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

np.set_printoptions(suppress=True)

class Neuron():
    def __init__(self, set, method):
        
        self.epochs = 10000
        self.epsilon = 0.0001
        self.learningRate = 0.0001
        self.beta = 1
        self.set = set
        self.method = method

        self.w = np.random.randn(3)

        self.batchSize = 10
        setLabel = self.set[:,2]
        self.batchLabel = setLabel.reshape(-1, self.batchSize, 1)

        self.set = np.delete(self.set, 2, 1)
        self.set = np.c_[self.set, np.ones(np.shape(self.set)[0]) * -1]
        self.numberOfBatches = math.ceil(self.set.shape[0]/self.batchSize)
        self.batches = self.set.reshape(self.numberOfBatches, self.batchSize, 3)
        
    
    def train(self):
        for iteration in range(self.epochs):
            for i in range(self.numberOfBatches):
                self.output = self.batches[i] @ self.w
                activationFunc = self.methodPicker(self.output)
                error = np.subtract(self.batchLabel[i], activationFunc.reshape((self.batchSize, 1)))
                deltaW = self.learningRate * error * self.derivative(self.output).reshape(error.shape) * self.batches[i]
                finalDeltaW = np.mean(deltaW, axis=0)
                self.w = self.w + finalDeltaW
        
        
    def methodPicker(self, output):
        if self.method == 1:
            return self.logistic(output)
        elif self.method == 2:
            return np.sin(output)
        elif self.method == 3:
            return np.tanh(output)
        elif self.method == 4:
            return np.sign(output)
        elif self.method == 5:
            return self.relu(output)
        elif self.method == 6:
            return self.leakyRelu(output)
        else: # default method
            return np.heaviside(output, 0.5)
    
    def logistic(self, output):
        value = np.ones(output.shape)
        for i in range(output.size):
            value[i] = 1 / (1 + math.exp(output[i]*self.beta))
        return value

    def derivative(self, output):
        if self.method == 1:
            return self.logistic(output) * (1 - self.logistic(output))
        elif self.method == 2:
            return np.cos(output)
        elif self.method == 3:
            return (1 - np.tanh(output) * np.tanh(output))
        elif self.method == 4:
            return np.ones(output.shape)
        elif self.method == 5:
            return self.reluD(output)
        elif self.method == 6:
            return self.leakyReluD(output)
        else: #default is derivative for heaviside
            return np.ones(output.shape)
        
        
    def preparePointsToPredict(self, x, y):
        self.w = self.w.reshape((3,))
        x = x.reshape((x.size, 1))
        y = y.reshape((y.size, 1))
        data = np.c_[x, y]
        data = np.c_[data, np.ones(np.shape(data)[0])*-1]
        value = data @ self.w
        return self.methodPicker(value)
    
    def relu (self, output):
        for i in range(output.size):
            if output[i] > 0:
                output[i] = output[i]
            else:
                output[i] = 0
        return output

    def reluD (self, output):
        for i in range(output.size):
            if output[i] > 0:
                output[i] = 1
            else:
                output[i] = 0
        return output

    def leakyRelu(self, output):
        for i in range(output.size):
            if output[i] > 0:
                output[i] = output[i]
            else:
                output[i] = 0.01 * output[i]
        return output

    def leakyReluD(self, output):
        for i in range(output.size):
            if output[i] > 0:
                output[i] = 1
            else:
                output[i] = 0.01 * output[i]
        return output
    
class Plot(FigureCanvasQTAgg):
    def __init__(self, parent = None):
        self.plot = plt.figure()
        self.scatter = self.plot.add_subplot(111)
        super().__init__(self.plot)
        self.setParent(parent)
        self.createPlot(50, 2)
    
    def createSet(self, class0Samples, class1Samples):
        class0_temp = class0Samples[0]
        class1_temp = class1Samples[0]
        for i in range(1, class0Samples.shape[0]):
            class0_temp = np.r_[class0_temp, class0Samples[i]]
            class1_temp = np.r_[class1_temp, class1Samples[i]]
        
        shape0 = class0_temp.shape[0] + class1_temp.shape[0]
        temp = np.zeros((shape0, 3))
        for x in range(class0_temp.shape[0]):
            temp[x] = np.array([class0_temp[x][0], class0_temp[x][1], 0])
            temp[x+int(shape0/2)] = np.array([class1_temp[x][0], class1_temp[x][1], 1])

        np.random.shuffle(temp)
        self.set = temp
        
    def getSet(self):
        return self.set
    
    def createPlot(self, samplesNumber = 50, modesNumber = 2):
        self.plot.clear()
        self.scatter = self.plot.add_subplot(111)
        class0_modes = np.random.rand(modesNumber, 2)
        class1_modes = np.random.rand(modesNumber, 2)
        class0Samples = np.zeros((modesNumber, samplesNumber, 2))
        class1Samples = np.zeros((modesNumber, samplesNumber, 2))
        
        for i in range(modesNumber):
            class0Samples[i] = np.random.normal(loc = class0_modes[i], scale = 0.1, size = (samplesNumber, 2))
            class1Samples[i] = np.random.normal(loc = class1_modes[i], scale = 0.1, size = (samplesNumber, 2))

        self.scatter.scatter(class0_modes[:, 0], class0_modes[:, 1], c = 'red', marker = "P")
        self.scatter.scatter(class0Samples[:, :, 0], class0Samples[:, :, 1], c = 'red', marker = '*')
        self.scatter.scatter(class1_modes[:, 0], class1_modes[:, 1], c = 'blue', marker = "P")
        self.scatter.scatter(class1Samples[:, :, 0], class1Samples[:, :, 1], c = 'blue', marker = '*')
        
        self.class0Samples = class0Samples
        self.class1Samples = class1Samples
        
        self.createSet(class0Samples, class1Samples)
        self.getMaxes()
        self.scatter.set_xlim(self.minX, self.maxX)
        self.scatter.set_ylim(self.minY, self.maxY)
    
    def getMaxes(self):
        self.minX = np.amin(np.hstack(self.set[:,0])) * 1.1
        self.maxX = np.amax(np.hstack(self.set[:,0])) * 1.1
        
        self.minY = np.amin(np.hstack(self.set[:,1])) * 1.1
        self.maxY = np.amax(np.hstack(self.set[:,1])) * 1.1
        
    def createBoundary(self): 
        
        xAxis = np.linspace(self.minX, self.maxX, 50)
        yAxis = np.linspace(self.minY, self.maxY, 50)
        xVals, yVals = np.meshgrid(xAxis, yAxis)
        return xVals, yVals

    def createPlotWithBoundary(self, prediction, x, y, method):
        self.plot.clear()
        self.scatter = self.plot.add_subplot(111)
        self.scatter.set_xlim(self.minX, self.maxX)
        self.scatter.set_ylim(self.minY, self.maxY)
        self.scatter.contourf(x, y, prediction, levels=1, cmap=cm.gray)
        
        '''temp = np.c_[x.reshape(2500,), y.reshape(2500,), prediction.reshape(2500,)]
        temp0 = []
        temp1 = []
        if method in [0, 1, 5]:
            for x in range(temp.shape[0]):
                if temp[x][2] > 0.5:
                    temp0.append([temp[x][0], temp[x][1]])
                elif temp[x][2] <= 0.5:
                    temp1.append([temp[x][0], temp[x][1]])
        else:
            for x in range(temp.shape[0]):
                if temp[x][2] > 0:
                    temp0.append([temp[x][0], temp[x][1]])
                elif temp[x][2] <= 0:
                    temp1.append([temp[x][0], temp[x][1]])
        temp0 = np.array(temp0)
        temp1 = np.array(temp1)
        try:
            self.scatter.scatter(temp0[:, 0], temp0[:, 1], c = '#c7c7f0', marker = 'o', s=100)
        except:
            print('all points predicted as 1 value')
        try:
            self.scatter.scatter(temp1[:, 0], temp1[:, 1], c = '#ebcac7', marker = 'o', s=100)
        except:
            print('all points predicted as 1 value')'''
        
        self.scatter.scatter(self.class0Samples[:, :, 0], self.class0Samples[:, :, 1], c = 'red', marker = '*')
        self.scatter.scatter(self.class1Samples[:, :, 0], self.class1Samples[:, :, 1], c = 'blue', marker = '*')
        
        
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.plot = Plot(self)
        self.setGeometry(50,50,900,480)
        
        onlyInt = QIntValidator()
        onlyInt.setRange(1, 1000)

        self.modesLabel = QLabel("number of modes", self)
        self.modesLabel.setGeometry(710, 30, 100, 20)
        
        self.numberOfModesWidget = QLineEdit(self)
        self.numberOfModesWidget.setText("2")
        self.numberOfModesWidget.setGeometry(810, 30, 50, 20)
        self.numberOfModesWidget.setValidator(onlyInt)
        
        self.samplesLabel = QLabel("number of samples", self)
        self.samplesLabel.setGeometry(710, 60, 100, 20)
        
        self.numberOfSamplesWidget = QLineEdit(self)
        self.numberOfSamplesWidget.setText("50")
        self.numberOfSamplesWidget.setGeometry(810, 60, 50, 20)
        self.numberOfSamplesWidget.setValidator(onlyInt)
        
        self.createPlotButton = QPushButton(self)
        self.createPlotButton.setText("create plot")
        self.createPlotButton.setGeometry(720, 90, 130, 20)
        self.createPlotButton.clicked.connect(lambda: self.createPlot())
        
        self.methodWidget = QListWidget(self)
        self.methodWidget.addItem('Heaviside')
        self.methodWidget.addItem('Logistic')
        self.methodWidget.addItem('Sin')
        self.methodWidget.addItem('Tanh')
        self.methodWidget.addItem('Sign')
        self.methodWidget.addItem('Relu')
        self.methodWidget.addItem('LRelu')
        self.methodWidget.setGeometry(735, 120, 100, 140)
        
        self.neuronButton = QPushButton(self)
        self.neuronButton.setText("Neuron")
        self.neuronButton.setGeometry(720, 270, 130, 20)
        self.neuronButton.clicked.connect(lambda: self.createNeuron())
        
        self.show()

    def createPlot(self):
        self.plot.createPlot(modesNumber=int(self.numberOfModesWidget.text()), 
                             samplesNumber=int(self.numberOfSamplesWidget.text()))
        self.plot.draw()
        
    def createNeuron(self):
        self.neuron = Neuron(self.plot.getSet(), int(self.methodWidget.currentRow()))
        self.neuron.train()
        self.xVals, self.yVals = self.plot.createBoundary()
        self.predictedLabels = self.neuron.preparePointsToPredict(self.xVals, self.yVals)
        self.predictedLabels = self.predictedLabels.reshape((50,50))
        
        self.plot.createPlotWithBoundary(self.predictedLabels, self.xVals, self.yVals, int(self.methodWidget.currentRow()))
        self.plot.draw()
        

def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()

    app.exec()
    
if __name__ == "__main__":
    main()
        