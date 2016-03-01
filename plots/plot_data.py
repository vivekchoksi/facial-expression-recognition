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

def main():
    path = parse_inputs()
    results = read_files(path)
    plot_cross_val(results)

if __name__ == '__main__':
  main()
