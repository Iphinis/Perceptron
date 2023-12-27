import numpy as np
import matplotlib.pyplot as plt

class SingleLayerPerceptron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.full(input_size, 0.5)
        self.bias = np.random.rand()

    def activation_function(self, s):
        return 1 if s >= 0 else 0

    def predict(self, x, debug=False):
        s = np.dot(x, self.weights) + self.bias
        
        res = self.activation_function(s)

        if debug:
            print("Activation for", x, "is", res)

        return res
        
    def minmax_scaling(self, data, xmin=None, xmax=None, debug=False):
        bounds = True
        if xmin is None and xmax is None:
            xmin = np.min(data)
            xmax = np.max(data)
            bounds = False
        
        normalized_data = (data - xmin)/(xmax - xmin)

        if debug:
            print("--- minmax_scaling DEBUG ---")

            print("data:\n", data)
            print("min:", xmin, "; max:", xmax)
            print("normalized_data:\n", normalized_data)

            print("--- END OF minmax_scaling DEBUG ---")
        

        if not bounds:
            return normalized_data, xmin, xmax
        else:
            return normalized_data

    def train(self, input_data, labels, learning_rate=0.1, epochs=100, stats = False):
        if stats:
            meanerrors = []
            true_positive, false_positive, false_negative, true_negative = (0,0,0,0)

        for epoch in range(epochs):
            if stats:
                errors = []
            for(x, label) in zip(input_data, labels):
                prediction = self.predict(x)
                error = label - prediction

                self.weights += learning_rate * error * x
                self.bias += learning_rate * error

                if stats:
                    errors.append(abs(error))

                    # negative class' label : 0
                    # positive class' label : 1

                    if prediction == 1:
                        if prediction == label:
                            true_positive += 1
                        else:
                            false_positive += 1
                    elif prediction == 0:
                        if prediction == label:
                            true_negative += 1
                        else:
                            false_negative += 1

            if stats:
                meanerrors.append(np.mean(errors))
        
        if stats:
            return np.array(meanerrors), true_positive, false_positive, false_negative, true_negative
        
    def get_params(self):
        return self.bias, self.weights

    def show_params(self):
        print("Weights:", self.weights)
        print("Bias:", self.bias)

    def get_predictions(self, points, debug=False):
        labels = []

        for point in points:
            res = self.predict(point, debug=debug)
            labels.append(res)
        
        return np.array(labels)
