from network import *
import random

testNetwork = Network([3, 100, 8])

trainingInputs = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]

trainingOutputs = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
]

for i in range(1000):

    sampleInputs = []
    sampleExpecteds = []

    for i in range(16):
        selection = random.randint(0, 7)
        sampleInputs.append(trainingInputs[selection])
        sampleExpecteds.append(trainingOutputs[selection])

    testNetwork.train(sampleInputs, sampleExpecteds, 0.01)

    print("------- STARTING TEST BATCH --------")

    for i in range(8):
        testNetwork.generate_output(trainingInputs[i])
        print(testNetwork.get_output())

        if testNetwork.get_output().index(max(testNetwork.get_output())) == i:
            print("Correct!")

        else:
            print("Incorrect")

for i in range(1000):

    sampleInputs = []
    sampleExpecteds = []

    for i in range(16):
        selection = random.randint(0, 7)
        sampleInputs.append(trainingInputs[selection])
        sampleExpecteds.append(trainingOutputs[selection])

    testNetwork.train(sampleInputs, sampleExpecteds, 0.005)

    print("------- STARTING TEST BATCH --------")

    for i in range(8):
        testNetwork.generate_output(trainingInputs[i])
        print(testNetwork.get_output())

        if testNetwork.get_output().index(max(testNetwork.get_output())) == i:
            print("Correct!")

        else:
            print("Incorrect")