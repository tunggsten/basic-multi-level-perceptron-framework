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



class Neuron:
    def __init__(self, weights:list[float], bias:float):
        self.weights = weights
        self.bias = bias

        self.weightedInput = 0
        self.activation = 0

        self.previousLayer = None
        self.nextLayer = None

        self.neuronIndex = None

        self.dCostByDWeights = []
        self.dCostByDBias = 0

        self.dCostByDActivation = 0

    def export_parameters(self):
        return(f"{self.weights}\\{self.bias}\n")

    def get_derivatives(self):
        return f"Derivative of cost with respect to weights: \n{self.dCostByDWeights}\nDerivative of cost with respect to bias: \n{self.dCostByDBias}\nDerivative of cost with respect to activation:\n{self.dCostByDActivation}"


    def calculate_activation(self, inputs:list[float]):
        self.weightedInput = 0
        
        for i in range(len(inputs)):
            self.weightedInput += self.weights[i] * inputs[i]

        self.weightedInput += self.bias

        self.activation = l_ReLU(self.weightedInput)



    def get_d_weighted_input_by_d_weight(self, weightIndex:int):
        if self.previousLayer:
            return self.previousLayer.neurons[weightIndex].activation
    
    def get_d_activation_by_d_weighted_input(self):
        return 1 if self.weightedInput >= 0 else LRELULEAKSLOPE
    


    def calculate_derivatives(self):
        self.dCostByDWeights = []
        self.dCostByDBias = 0
        
        dActivationByDInput = neuron.get_d_activation_by_d_weighted_input()
        
        # Find d activation by d cost
        # This is helpful because it lets us collapse our derivative chain for each layer of the network,
        # so we can process the preceeding layers without redoing big calculations
        
        if self.nextLayer:
            self.dCostByDActivation = 0
            
            for neuron in self.nextLayer:
                self.dCostByDActivation += (neuron.weights[self.neuronIndex] * 
                                            dActivationByDInput * 
                                            neuron.dCostByDActivation)

        if self.previousLayer:
            # Find d cost by d weight for all weights
            for i in range(self.previousLayer.size):
                self.dCostByDWeights.append(self.previousLayer.neurons[i].activation *
                                            dActivationByDInput *
                                            self.dCostByDActivation)
                
            # d weighted input by d bias is always 1, so we don't have to multiply anything here
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
            
            for i in range(self.previousLayer.size):
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

        self.layers.append(Layer(layerSizes[0]))

        for i in range(1, self.layerCount):
            layer = Layer(layerSizes[i])

            self.layers.append(layer)

            layer.previousLayer = self.layers[i - 1]
            layer.initialise_neuron_weights()
            
        for i in range(self.layerCount - 2):
            self.layers[i].nextLayer = self.layers[i + 1]

        self.inputLayer = self.layers[0]
        self.outputLayer = self.layers[self.layerCount - 1]



    def export_model(self):
        output = ""

        for layer in self.layers:
            output += layer.export_layer_model()

        return output
    
    def save_model_to_file(self, filename:str):
        if filename[-6 :] != ".model":
            print(f"{filename} isn't a valid .model file. Double check you're trying to save this to the right path")

        choice = input(f"\n--- WARNING --- \n You are attempting to save a model. This will permenantly overwrite everything in {filename}. Are you sure? (yes/no)\n")

        if choice.lower() == "yes":
            with open(filename, "w") as f:
                f.write(self.export_model())
            
            print("Saved!")
    


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
    
    

    def get_output(self):
        return self.outputLayer.get_layer_activation()

    def generate_output(self, inputs:list[float]):
        self.outputLayer.calculate_model_outputs(inputs)


    
    def get_ssr(self, expected:list[float]):
        outputs = self.outputLayer.get_layer_activation()

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

            for line in f.readlines():
                if line[0] == "l":
                    info = line.split("\\")

                    network.layers.append(Layer(0))
                    topLayer += 1

                else:
                    info = line.split("\\")

                    network.layers[topLayer].neurons.append(Neuron(ast.literal_eval(info[0]), float(info[1])))

            network.layerCount = topLayer + 1

            for i in range(1, network.layerCount):
                layer = network.layers[i]

                layer.previousLayer = network.layers[i - 1]
                
            for i in range(network.layerCount - 2):
                network.layers[i].nextLayer = network.layers[i + 1]

            network.inputLayer = network.layers[0]
            network.outputLayer = network.layers[network.layerCount - 1]

            return network
    except:
        print("Could not interpret model data; your model file may be corrupted or a model file by this name couldn't be found.")


        
startTime = time.time()
testNetwork = generate_network_from_model("testModel.model")
print(f"Loaded model in {time.time() - startTime} seconds.")

startTime = time.time()
testNetwork.generate_output([2.3, 1, 5])
timeMessage = f"Finished processing model in {time.time() - startTime} seconds."

print(testNetwork.outputLayer.get_layer_activation())

print(timeMessage)

print("Starting backpropagation...")
startTime = time.time()
testNetwork.backpropagate([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
print(f"Finished backpropagation in {time.time() - startTime} seconds.")

print(testNetwork.get_network_derivatives([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))