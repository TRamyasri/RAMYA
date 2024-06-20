import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output

def forward_propagation(inputs, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

def backpropagation(inputs, hidden_layer_output, output_layer_output, expected_output, weights_input_hidden, weights_hidden_output, learning_rate):
    output_layer_error = expected_output - output_layer_output
    output_layer_delta = output_layer_error * sigmoid_derivative(output_layer_output)

    hidden_layer_error = output_layer_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(output_layer_delta) * learning_rate
    weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate

    return weights_input_hidden, weights_hidden_output

def train(inputs, expected_output, input_size, hidden_size, output_size, learning_rate, epochs):
    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output)
        weights_input_hidden, weights_hidden_output = backpropagation(inputs, hidden_layer_output, output_layer_output, expected_output, weights_input_hidden, weights_hidden_output, learning_rate)

        if epoch % 1000 == 0:
            loss = np.mean(np.square(expected_output - output_layer_output))
            print(f'Epoch {epoch} - Loss: {loss}')

    return weights_input_hidden, weights_hidden_output

def predict(inputs, weights_input_hidden, weights_hidden_output):
    _, output_layer_output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output)
    return output_layer_output

# Example usage
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 10000

weights_input_hidden, weights_hidden_output = train(inputs, expected_output, input_size, hidden_size, output_size, learning_rate, epochs)
predictions = predict(inputs, weights_input_hidden, weights_hidden_output)
print("Predictions:")
print(predictions)
