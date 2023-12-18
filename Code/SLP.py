import numpy as np

class SingleLayerPerceptron:
    def __init__(self, input_size):
        self.weights = np.full(input_size, 0.5)
        self.bias = np.random.rand()

    def activation_function(self, s):
        return 1 if s >= 0 else 0

    def predict(self, x):
        s = np.dot(x, self.weights) + self.bias
        return self.activation_function(s)

    def train(self, input_data, labels, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            for(data, label) in zip(input_data, labels):
                prediction = self.predict(data)
                error = label - prediction
                self.weights += learning_rate * error * data
                self.bias += learning_rate * error

    def show_params(self):
        print("Bias:", self.bias)
        print("Weights:", self.weights)

if __name__ == "__main__":
    training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 0, 1])

    SLP = SingleLayerPerceptron(2)

    SLP.train(training_data, labels)

    SLP.show_params()

    print(SLP.predict([1,1]))
