# Read in log files
# Visualize the cross-validation results

import glob
import os
import sys
import math
import matplotlib.pyplot as plt
import argparse

def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default = '../outputs', help = 'path to the folder containing the log files')
    args = parser.parse_args()
    return args.f

def read_files(path):
    results = {}

    for filename in glob.glob(os.path.join(path, '*.txt')):        
        print "reading " +  filename + "..."

        for row in open(filename, 'rb'):
            line = (row.strip().split(':'))
            if line[0] == 'acc':
                acc_array = line[1].strip().split(',')
                acc = float(acc_array[-1].strip().split(']')[0])
                print "accuracy: ", acc
            elif line[0] == 'val_acc':
                val_acc_array = line[1].strip().split(',')
                val_acc = float(val_acc_array[-1].strip().split(']')[0])
                print "validation accuracy: ", val_acc
            elif line[0] == 'lr':
                lr = float(line[1].strip()) 
            elif line[0] == 'reg':
                reg = float(line[1].strip()) 
            elif line[0] == 'dropout':
                dropout = float(line[1].strip()) 
        results[(lr,reg)] = (acc, val_acc)
    return results

def plot_cross_val(results):
    # Visualize the cross-validation results
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    # plot validation accuracy
    colors = [results[x][1] for x in results] # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show()

def plot_dropout_comparison():
    x = xrange(1, 11)

    # The train and validation accuracies of models with low and high dropout parameters.
    y_high_train = [0.2634713852798774, 0.38879793792887246, 0.43756313351213905, 0.46797171618656169, 0.49381726984569296, 
        0.51307952210108332, 0.52927653349123971, 0.53934306315092828, 0.5474938172698457, 0.55421644780382462]
    y_high_val = [0.25327389244915016, 0.28810253552521592, 0.18333797715241015, 0.3145723042630259, 0.35051546391752575,
        0.35748119253273891, 0.38757314015045974, 0.3976037893563667, 0.40039008080245192, 0.42574533296182782]
    y_low_train = [0.34212267929917445, 0.45355115120693856, 0.50440628374377372, 0.53972621825908251, 0.56020760040405448,
        0.57804172907450624, 0.59869727263227557, 0.61235152739559018, 0.62579678846354803, 0.6376049322512104]
    y_low_val = [0.3455001393145723, 0.39593201448871551, 0.39955419336862635, 0.47199777096684314, 0.45611590972415716,
        0.43661186960156034, 0.45360824742268041, 0.46391752577319589, 0.470883254388409, 0.44469211479520759]
    plt.plot(x, y_high_train, 'r--', label='train acc with higher dropout')
    plt.plot(x, y_high_val, 'r-', label='val acc with higher dropout')
    plt.plot(x, y_low_train, 'b--', label='train acc with lower dropout')
    plt.plot(x, y_low_val, 'b-', label='val acc with lower dropout')

    plt.title('Train and validation accuracy with varying dropout')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()

def main():
    path = parse_inputs()
    results = read_files(path)
    plot_cross_val(results)
    # plot_dropout_comparison()

if __name__ == '__main__':
  main()
