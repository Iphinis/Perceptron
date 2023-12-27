import numpy as np
import matplotlib.pyplot as plt

def get_dataset(filename, sep, data_cols, label_col, skiprows, debug=False):

    usecols = np.union1d(data_cols, label_col)
    dataset = np.loadtxt(filename, delimiter=sep, usecols=usecols, skiprows=skiprows)

    data = dataset[:, data_cols]
    labels = dataset[:, label_col]

    if debug:
        print("--- get_dataset(" + filename + ") DEBUG ---")
        print("used cols for dataset:", usecols)
        print("dataset:\n", dataset)

        print("used cols for data:", data_cols)
        print("data:\n", data)

        print("used col for labels:", label_col)
        print("labels\n:", labels)

        print("--- END OF get_dataset(" + filename + ") DEBUG ---")

    return data, labels

def show_error(err, yscale="linear"):
    plt.plot(err)#, np.shape(err))
    plt.xlabel("Epochs")
    plt.ylabel("Mean error (" + yscale + " scale)")

    plt.yscale(yscale)

    plt.autoscale()

    plt.grid()
    plt.show()

def learning_stats(true_positive, false_positive, false_negative, true_negative):
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    accuracy = (true_positive + true_negative)/(true_positive + false_positive + false_negative + true_negative)

    return precision, recall, accuracy

def plot_2D_dataset(data, labels, bias=None, weights=None, pred_data=None, pred_labels=None):
    assert(np.shape(data)[1] == 2), "plot_2D_dataset error: You are trying to plot a non-2D dataset."
    assert np.all(np.logical_or(labels == 0, labels == 1)), "plot_2D_dataset error: labels must only be 0 or 1."
    if weights is not None:
    	assert(np.shape(weights)[0] == 2), "plot_2D_dataset error: weights are not exactly two"
    if pred_labels is not None:
    	assert np.all(np.logical_or(pred_labels == 0, pred_labels == 1)), "plot_2D_dataset error: pred_labels must only be 0 or 1."
    
    # separate points according to their label
    points_label_0 = data[labels == 0]
    points_label_1 = data[labels == 1]
    
    # scatter points
    plt.scatter(points_label_0[:, 0], points_label_0[:, 1], marker='o', color='red', label='Label 0')
    plt.scatter(points_label_1[:, 0], points_label_1[:, 1], marker='s', color='darkblue', label='Label 1')
    
    # plot decision boundary
    if bias is not None and weights is not None:
        x = np.linspace(np.min(data, axis=0)[0], np.max(data, axis=0)[0])

        w1, w2 = weights[0], weights[1]
        y = (-1/w2)*(w1*x + bias)

        plt.plot(x, y, linestyle='-', color='green', label='Decision Boundary')

    if pred_data is not None and pred_labels is not None:
        # separate predicted points according to their label
        points_pred_label_0 = pred_data[pred_labels == 0]
        points_pred_label_1 = pred_data[pred_labels == 1]
        
        # scatter predicted points
        plt.scatter(points_pred_label_0[:, 0], points_pred_label_0[:, 1], marker='o', color='orange', label='Label 0 (predicted)')
        plt.scatter(points_pred_label_1[:, 0], points_pred_label_1[:, 1], marker='s', color='cyan', label='Label 1 (predicted)')

    plt.legend()
    plt.grid()
    plt.show()
