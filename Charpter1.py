#!/usr/bin/env python
# coding: utf-8
import numpy
import scipy
from matplotlib import pyplot as plt
# sigmoid函数
# 输入:numpy列表
# 输出:sigmoid函数运算结果
def sigmoid(mat):
    # 确保输入为numpy数组
    mat = numpy.array(mat)
    return 1.0/(1+numpy.exp(-mat))
# 三层神经网络框架
class NeuralNetowrk:
    # 初始化函数
    # 输入:   输入层节点数，隐藏层节点数，输出层节点数，学习率
    # 输出：  None
    def __init__(self,input_node,hidden_node,output_node,learningrate):
        self.input_node = input_node
        self.hidden_node = hidden_node
        self.output_node = output_node
        self.learningrate = learningrate
        # 构建输入-隐藏层权重矩阵，行数为隐藏层节点数，列数为输入层节点数
        # 按照正太分布采样,生成范围在sqrt(n)之间
        self.weight_input_hidden = numpy.random.normal(0.0,pow(self.hidden_node,-0.5),(self.hidden_node,self.input_node))
        # 构建隐藏-输出层权重矩阵，行数为隐藏层节点数，列数为输出层节点数
        self.weight_hidden_output = numpy.random.normal(0.0,pow(self.output_node,-0.5),(self.hidden_node,self.output_node))
    # 训练神经网络,优化权重
    def train(self):
        pass
    # 查询神经网络，给入输入，获得输出
    # 输入:矩阵
    # 输出：矩阵
    def query(self,inputs):
        # 隐藏层输入
        hidden_input = numpy.dot(self.weight_input_hidden,inputs)
        # 隐藏层输出
        hidden_output = sigmoid(hidden_input)
        # 输出层输入
        final_input = numpy.dot(self.weight_hidden_output,hidden_output)
        # 输出层输出
        final_output = sigmoid(final_input)
        return final_output
if __name__=='__main__':
    MyNN = NeuralNetowrk(10,10,10,0.5)
    input = numpy.arange(-5,5,step=1)
    output = MyNN.query(input)
    print("Input : ",input)
    print(output)
    plt.figure(1)
    plt.plot(range(-10,10),sigmoid(range(-10,10)))
    plt.show()
    
