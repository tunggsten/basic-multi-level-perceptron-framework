from network import *

import linecache

def interpret_mnist(mnistDataIndex:int):
    try:
        data = linecache.getline("MNISTtrain.csv", mnistDataIndex + 1)
        data = data.split(",")
        
        inputs = []
        
        for element in data:
            inputs.append(int(element))
        
        expectedDigit = inputs.pop(0)
        
        expectedOutput = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        expectedOutput.insert(expectedDigit, 1)
        
        for i in range(len(inputs)):
            inputs[i] /= 255
            
        return [inputs, expectedOutput]
    except:
        return interpret_mnist(0)
    

choice = input("Welcome to the MNIST trainer. Would you like to start [F]resh, or load data from a file (type filename)? ")

if choice.lower() == "f":
    choice = input("Enter the file you'd like to save to: ")
    network = Network([784, 256, 256, 128, 10])
else:
    network = generate_network_from_model(choice)

attempts = []
previousLosses = []

for i in range(0, 540):
    
    # Perform training cycle
    trainingInputs = []
    trainingOutputs = []
    
    for j in range(75):
        data = interpret_mnist((i) * 75 + j)
        
        trainingInputs.append(data[0])
        trainingOutputs.append(data[1])
        
    network.train(trainingInputs, trainingOutputs, 0.01)
    
    print(f"\nFinished training cycle {i}.")
    
    network.save_model_to_file(choice)