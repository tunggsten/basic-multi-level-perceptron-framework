from network import *

import linecache

import pygame

pygame.init()

#display = pygame.display.set_mode((854, 480))
pygame.display.set_caption("MNIST")

pygame.font.init()

font = pygame.font.Font(None, 36)



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
        
        

def display_mnist(outputData:list[int], location:tuple=(16, 16), pixelSize:int=8):
    writtenDigit = pygame.Surface((pixelSize * 28, pixelSize * 28 + 16))
    
    for row in range(28):
        for collumb in range(28):
            colour = outputData[row * 28 + collumb] * 255
            pygame.draw.rect(writtenDigit, (colour, colour, colour), pygame.Rect(collumb * pixelSize, (row * pixelSize + 16), pixelSize, pixelSize))
    
    writtenDigit.blit(font.render("Input:", True, (255, 255, 255)), (0, 0))
                             
    display.blit(writtenDigit, location)
    
def display_expected(expected:list[int]):
    pass
    

choice = input("Welcome to the MNIST trainer. Would you like to start [F]resh, or load data from a file (type filename)? ")

if choice.lower() == "f":
    choice = input("Enter the file you'd like to save to: ")
    network = Network([784, 200, 100, 80, 10])
else:
    network = generate_network_from_model(choice)

attempts = []
previousLosses = []

for i in range(250):
    
    # Perform training cycle
    trainingInputs = []
    trainingOutputs = []
    
    for j in range(100):
        data = interpret_mnist((i) * 100 + j)
        
        trainingInputs.append(data[0])
        trainingOutputs.append(data[1])
        
    network.train(trainingInputs, trainingOutputs, 0.005)
    
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


'''
running = True

i = 0

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    testDigit = interpret_mnist(i)

    i += 1
            
    display.fill((50, 150, 255))
            
    display_mnist(testDigit[0])
        
    pygame.display.flip()'''