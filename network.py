# This is a basic attempt at making a neural network. I'll admit, 
# I don't really know what I'm doing but that's not going to stop me!

# I'll probably look at this in a few months and remember how stupid 
# I was for trying this

import math
import random
import time
import ast


LRELULEAKSLOPE = 0.1

def l_ReLU(input:float):
    return input if input >= 0 else LRELULEAKSLOPE * input

def sigmoid(input:float):
    return 1 / (1 + math.exp(-input))

def d_sigmoid_by_d_x(input:float):
    return sigmoid(input) * (1 - sigmoid(input))



class Neuron:
    def __init__(self, weights:list[float], bias:float):
        # Parameters
        self.weights = weights
        self.bias = bias

        # Processed values
        self.weightedInput = 0
        self.activation = 0

        # Layer indexes
        self.previousLayer = None
        self.nextLayer = None

        self.neuronIndex = None

        # Derivatives
        self.dCostByDWeights = []
        self.dCostByDBias = 0

        self.dCostByDActivation = 0

        # Average derivatives
        self.averageDCostByDWeights = []
        self.averageDCostByDBias = 0



    def export_parameters(self):
        return(f"{self.weights}\\{self.bias}\n")

    def get_derivatives(self):
        return (self.dCostByDWeights, self.dCostByDBias, self.dCostByDActivation)


    def calculate_activation(self, inputs:list[float]):
        self.weightedInput = 0
        for i in range(len(inputs)):
            self.weightedInput += self.weights[i] * inputs[i]

        self.weightedInput += self.bias

        self.activation = sigmoid(self.weightedInput)


    
    def reset_average_parameters(self):
        self.averageDCostByDBias = 0
        self.averageDCostByDWeights = []

        for i in range(len(self.weights)):
            self.averageDCostByDWeights.append(0)

    def add_derivatives_to_averages(self):
        self.averageDCostByDBias += self.dCostByDBias

        for i in range(len(self.weights)):
            self.averageDCostByDWeights[i] += self.dCostByDWeights[i]

    def scale_averages(self, sampleSize:int):
        self.averageDCostByDBias /= sampleSize

        for derivative in self.averageDCostByDWeights:
            derivative /= sampleSize

    

    def descend(self, learningRate:float):
        self.bias -= self.averageDCostByDBias * learningRate

        for i in range(len(self.weights)):
            self.weights[i] -= self.averageDCostByDWeights[i] * learningRate



    def get_d_weighted_input_by_d_weight(self, weightIndex:int):
        if self.previousLayer:
            return self.previousLayer.neurons[weightIndex].activation
    
    def get_d_activation_by_d_weighted_input(self):
        return d_sigmoid_by_d_x(self.weightedInput)
    


    def calculate_derivatives(self):
        #print(f"\nCalculating derivatives for neuron {self.neuronIndex}")
        self.dCostByDWeights = []
        self.dCostByDBias = 0
        
        dActivationByDInput = self.get_d_activation_by_d_weighted_input()
        
        # Find d activation by d cost
        # This is helpful because it lets us collapse our derivative chain for each layer of the network,
        # so we can process the preceeding layers without redoing big calculations
        
        if self.nextLayer:
            self.dCostByDActivation = 0
            
            for neuron in self.nextLayer.neurons:
                self.dCostByDActivation += (neuron.weights[self.neuronIndex] * 
                                            dActivationByDInput * 
                                            neuron.dCostByDActivation)
                
            #print(f"d cost by d activation is {self.dCostByDActivation}")

        if self.previousLayer:
            # Find d cost by d weight for all weights
            #print(f"Previous layer size: {self.previousLayer.size}")
            for i in range(self.previousLayer.size):
                #print(f"d cost by d weight {i} is {self.previousLayer.neurons[i].activation *dActivationByDInput * self.dCostByDActivation}")
                
                self.dCostByDWeights.append(self.previousLayer.neurons[i].activation *
                                            dActivationByDInput *
                                            self.dCostByDActivation)
                
            # d weighted input by d bias is always 1, so we don't have to multiply anything here
            #print(f"d cost by d bias is {self.get_d_activation_by_d_weighted_input() * self.dCostByDActivation}")
            self.dCostByDBias = self.get_d_activation_by_d_weighted_input() * self.dCostByDActivation



class Layer:
    def __init__(self, size:int):
        self.size = size

        self.neurons = []

        for i in range(size):
            self.neurons.append(Neuron([1], 0))

        self.previousLayer = False
        self.nextLayer = False

    def initialise_neuron_weights(self):
        for i in range(self.size):
            self.neurons[i].previousLayer = self.previousLayer
            self.neurons[i].nextLayer = self.nextLayer
            self.neurons[i].neuronIndex = i

            self.neurons[i].weights = []

            if self.previousLayer:
                for j in range(self.previousLayer.size):
                    self.neurons[i].weights.append(random.normalvariate())



    def export_layer_model(self):
        output = f"l\\{self.size}\n"

        for neuron in self.neurons:
            output += neuron.export_parameters()

        return output
    


    def get_layer_activation(self):
        activations = []

        for neuron in self.neurons:
            activations.append(neuron.activation)

        return activations
    
    def get_layer_derivatives(self):
        derivatives = []
        
        for neuron in self.neurons:
            derivatives.append(neuron.get_derivatives())
            
        return derivatives
    


    def reset_average_parameters(self):
        for neuron in self.neurons:
            neuron.reset_average_parameters()

    def add_derivatives_to_averages(self):
        for neuron in self.neurons:
            neuron.add_derivatives_to_averages()

    def scale_averages(self, sampleSize:int):
        for neuron in self.neurons:
            neuron.scale_averages(sampleSize)



    def descend(self, learningRate:float):
        for neuron in self.neurons:
            neuron.descend(learningRate)
    
    

    def calculate_model_outputs(self, modelInputs:list[float]):
        if self.previousLayer:
            self.previousLayer.calculate_model_outputs(modelInputs)

            layerInputs = self.previousLayer.get_layer_activation()

            for neuron in self.neurons:
                neuron.calculate_activation(layerInputs)

        else:

            for i in range(len(modelInputs)):
                self.neurons[i].activation = modelInputs[i]
                
    def backpropagate(self):
        for neuron in self.neurons:
            neuron.calculate_derivatives()
            
        if self.previousLayer:
            self.previousLayer.backpropagate()



class Network:
    def __init__(self, layerSizes:list[int]):
        self.layers = []
        self.layerCount = len(layerSizes)

        for i in range(self.layerCount):
            layer = Layer(layerSizes[i])
            self.layers.append(layer)

        self.initialise_layer_parameters()

    def initialise_layer_parameters(self):
        print("Initialising layer parameters!")
        self.layerCount = len(self.layers)
        
        for i in range(1, self.layerCount):
            print(f"Setting up layer {i}'s previous layer")
            self.layers[i].previousLayer = self.layers[i - 1]
            
        for i in range(self.layerCount - 1):
            self.layers[i].nextLayer = self.layers[i + 1]

        for layer in self.layers:
            layer.size = len(layer.neurons)
            layer.initialise_neuron_weights()

        self.inputLayer = self.layers[0]
        self.outputLayer = self.layers[self.layerCount - 1]

        for layer in self.layers:
            print(f"{layer} has previous layer {layer.previousLayer} and next layer {layer.nextLayer}")


    def export_model(self):
        output = ""

        for layer in self.layers:
            output += layer.export_layer_model()

        return output
    
    def save_model_to_file(self, filename:str):
        if filename[-6 :] != ".model":
            print(f"{filename} isn't a valid .model file. Double check you're trying to save this to the right path")
            
        open(filename, 'w').close()

        with open(filename, "w+") as f:
            f.write(self.export_model())
            
            print("Saved model!")
    


    def get_network_activations(self):
        activations = []

        for layer in self.layers:
            activations.append(layer.get_layer_activation())

        return activations
    
    def get_network_derivatives(self):
        derivatives = []
        
        for layer in self.layers:
            derivatives.append(layer.get_layer_derivatives())
            
        return derivatives
    


    def reset_average_parameters(self):
        for layer in self.layers:
            layer.reset_average_parameters()

    def add_derivatives_to_averages(self):
        for layer in self.layers:
            layer.add_derivatives_to_averages()

    def scale_averages(self, sampleSize:int):
        for layer in self.layers:
            layer.scale_averages(sampleSize)



    def descend(self, learningRate:float):
        for layer in self.layers:
            layer.descend(learningRate)
    
    

    def get_output(self):
        return self.outputLayer.get_layer_activation()

    def generate_output(self, inputs:list[float]):
        self.outputLayer.calculate_model_outputs(inputs)


    
    def get_ssr(self, expected:list[float]):
        outputs = self.get_output()

        SSR = 0

        for i in range(len(outputs)):
            SSR += (outputs[i] - expected[i]) ** 2

        return SSR
    
    
    
    def backpropagate(self, expected:list[float]):
        # This finds the derivatives of each parameter with respect to the cost when the network's 
        # current activations are expected to return the expected values
        
        for i in range(self.outputLayer.size):
            outputNeurons = self.outputLayer.neurons
            outputNeurons[i].dCostByDActivation = -2 * (expected[i] - outputNeurons[i].activation)
            
        self.outputLayer.backpropagate()


    
    def train(self, sampleInputs:list[list[float]], sampleExpecteds:list[list[float]], learningRate:float):
        sampleSize = len(sampleInputs)

        self.reset_average_parameters()

        for i in range(sampleSize):
            self.generate_output(sampleInputs[i])
            self.backpropagate(sampleExpecteds[i])
            self.add_derivatives_to_averages()

        self.scale_averages(sampleSize)

        self.descend(learningRate)



def generate_network_from_model(model:str):
    print(f"Loading model from {model}...")

    if model[-6 :] != ".model":
        print("This may not be a .model file. Aborting")
        return

    network = Network([1, 1])

    network.layers = []

    try:
        with open(model, "r") as f:
            topLayer = -1
            topNeuron = -1

            for line in f.readlines():
                if line[0] == "l":
                    info = line.split("\\")

                    network.layers.append(Layer(0))
                    topLayer += 1
                    topNeuron = -1

                else:
                    info = line.split("\\")

                    topNeuron += 1
                    
                    network.layers[topLayer].neurons.append(Neuron(ast.literal_eval(info[0]), float(info[1])))
                    network.layers[topLayer].neurons[topNeuron].neuronIndex = topNeuron

        
        print("Initialising layer parameters!")
        network.layerCount = len(network.layers)
        
        for i in range(1, network.layerCount):
            print(f"Setting up layer {i}'s previous layer")
            network.layers[i].previousLayer = network.layers[i - 1]
            
        for i in range(network.layerCount - 1):
            network.layers[i].nextLayer = network.layers[i + 1]

        for layer in network.layers:
            layer.size = len(layer.neurons)

        network.inputLayer = network.layers[0]
        network.outputLayer = network.layers[network.layerCount - 1]

        for layer in network.layers:
            
            for i in range(layer.size):
                layer.neurons[i].previousLayer = layer.previousLayer
                layer.neurons[i].nextLayer = layer.nextLayer
                layer.neurons[i].neuronIndex = i

        return network
    except:
        print("Could not interpret model data; your model file may be corrupted or a model file by this name couldn't be found.")
