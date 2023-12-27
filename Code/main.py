import numpy as np
from SLP import SingleLayerPerceptron
from analysis import *

def main():
    # linear dataset
    print("--- LINEAR dataset ---")
    
    data, labels = get_dataset(filename="2class-lin-dataset.csv", sep=",", data_cols=np.arange(0,2), label_col=2, skiprows=1)

    perceptron = SingleLayerPerceptron(input_size=np.shape(data)[1])

    normalized_data, xmin, xmax = perceptron.minmax_scaling(data)

    err, true_positive, false_positive, false_negative, true_negative = perceptron.train(normalized_data, labels, learning_rate=0.01, epochs=75, stats=True)
    perceptron.show_params()

    precision, recall, accuracy = learning_stats(true_positive, false_positive, false_negative, true_negative)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Accuracy: ", accuracy)

    show_error(err)
    
    plot_2D_dataset(normalized_data, labels)

    pred_data = np.array([[-160, -101], [-100, -160], [200, 100]])
    
    normalized_pred_data = perceptron.minmax_scaling(pred_data, xmin, xmax)

    pred_labels = perceptron.get_predictions(normalized_pred_data)

    bias, weights = perceptron.get_params()
    
    plot_2D_dataset(normalized_data, labels, bias, weights, normalized_pred_data, pred_labels)



    # non linear dataset
    print("--- NON LINEAR dataset ---")
    
    data, labels = get_dataset(filename="2class-nonlin-dataset.csv", sep=",", data_cols=np.arange(0,2), label_col=2, skiprows=1)

    normalized_data, xmin, xmax = perceptron.minmax_scaling(data)

    err, true_positive, false_positive, false_negative, true_negative = perceptron.train(normalized_data, labels, learning_rate=0.01, epochs=75, stats=True)
    perceptron.show_params()

    precision, recall, accuracy = learning_stats(true_positive, false_positive, false_negative, true_negative)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Accuracy: ", accuracy)

    show_error(err)
    
    plot_2D_dataset(normalized_data, labels)

    bias, weights = perceptron.get_params()

    plot_2D_dataset(normalized_data, labels, bias, weights)



main()
