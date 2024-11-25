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
    network = Network([784, 250, 150, 80, 30, 10])
else:
    network = generate_network_from_model(choice)

attempts = []
previousLosses = []

for i in range(400):
    
    # Perform training cycle
    trainingInputs = []
    trainingOutputs = []
    
    for j in range(100):
        data = interpret_mnist((i) * 50 + j)
        
        trainingInputs.append(data[0])
        trainingOutputs.append(data[1])
        
    network.train(trainingInputs, trainingOutputs, 0.01)
    
    print(f"\nFinished training cycle {i}.")
    
    attempts = []
    testLosses = []
    # Run test case
    for k in range(30):
        testData = interpret_mnist(30 * i + k)

        network.generate_output(testData[0])
        output = network.get_output()
        
        loss = network.get_ssr(testData[1])
        
        guess = output.index(max(output))
        actual = testData[1].index(max(testData[1]))
        
        # Get average accuracy accross previous 100 attempts
        attempts.append(1 if guess == actual else 0)
        testLosses.append(loss)
            
        averageAccuracy = sum(attempts) / len(attempts)
        averageLoss = sum(testLosses) / len(testLosses)
        
        print(f"Prediction: {output} Expected: {testData[1]}. \nLoss is {loss}")
        print(f"Predicted {guess}, expected {actual}. {"\nCorrect!" if guess == actual else f"\nIncorrect"}\n")
        
    print(f"Average accuracy of {averageAccuracy * 100}%")
    print(f"Average loss of {averageLoss}")
    
    #input = input("Completed cycle. Do you want to save?")
    #if input.lower() == "yes":
    
    network.save_model_to_file(choice)