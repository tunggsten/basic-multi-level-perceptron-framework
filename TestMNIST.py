# This is a simple interactive demo of the MNIST model.

from network import *
import math
import linecache

import pygame

# initialise pygame
pygame.init()

window = pygame.display.set_mode((480, 360))

font = pygame.font.SysFont(None, 30)

backgroundColour = (50, 150, 255)

# initialise MASON
mason = generate_network_from_model("newNewMNIST.model")

class mnistDigit:
    def __init__(self, position:tuple):
        self.contents = []
        
        for i in range(784):
            self.contents.append(0)
            
        self.position = (0, 0)

    def render_input(self):
        inputSurface = pygame.Surface((244, 244))
        
        inputSurface.fill(backgroundColour)
        
        for row in range(28):
            for collumb in range(28):
                
                colour = math.floor(self.contents[row * 28 + collumb] * 255)
                pygame.draw.rect(inputSurface, (colour, colour, colour), pygame.Rect(collumb * 8, row * 8, 8, 8))
                
        window.blit(inputSurface, self.position)
                
    def draw(self, brushLocation, brushSize, brushPower):
        brushLocationRelative = (brushLocation[0] - self.position[0], brushLocation[1] - self.position[1])
        brushLocationRelative = (brushLocationRelative[0] / 8, brushLocationRelative[1] / 8)
        for row in range(28):
            for collumb in range(28):
                distance = math.sqrt((collumb - brushLocationRelative[0]) ** 2 + (row - brushLocationRelative[1]) ** 2)
                
                if distance < brushSize:
                    self.contents[row * 28 + collumb] += 1
                elif distance < brushPower:
                    self.contents[row * 28 + collumb] += (distance - brushPower) / 4 * (brushSize - brushPower)
                else:
                    self.contents[row * 28 + collumb] += 0
                    
                if self.contents[row * 28 + collumb] > 1:
                    self.contents[row * 28 + collumb] = 1
                    
    def erase(self, brushLocation, brushSize, brushPower):
        brushLocationRelative = (brushLocation[0] - self.position[0], brushLocation[1] - self.position[1])
        brushLocationRelative = (brushLocationRelative[0] / 8, brushLocationRelative[1] / 8)
        for row in range(28):
            for collumb in range(28):
                distance = math.sqrt((collumb - brushLocationRelative[0]) ** 2 + (row - brushLocationRelative[1]) ** 2)
                
                if distance < brushSize:
                    self.contents[row * 28 + collumb] -= 1
                elif distance < brushPower:
                    self.contents[row * 28 + collumb] -= (distance - brushPower) / 4 * (brushSize - brushPower)
                else:
                    self.contents[row * 28 + collumb] -= 0
                    
                if self.contents[row * 28 + collumb] < 0:
                    self.contents[row * 28 + collumb] = 0
                    
                    

neuronCount = len([x for xs in mason.get_network_activations()[1:] for x in xs])
                    
def display_mason_activations(position:tuple):
    outputSurface = pygame.Surface((256, 256))
    
    outputSurface.fill(backgroundColour)
    
    activations = [x for xs in mason.get_network_activations()[1:] for x in xs]
    
    for i in range(neuronCount):
        colour = math.floor(activations[i] * 255)
        
        pygame.draw.rect(outputSurface, (colour, colour, colour), pygame.Rect((i % 16) * 8, (i // 16) * 8, 8, 8))
        
    window.blit(outputSurface, position)
    
def display_mason_output(position:tuple):
    outputSurface = pygame.Surface((160, 32))
    
    outputSurface.fill(backgroundColour)
    
    output = mason.get_output()
    
    for i in range(10):
        colour = math.floor(output[i] * 255)
        pygame.draw.rect(outputSurface, (colour, colour, colour), (i * 16, 0, 16, 16))
        
    outputSurface.blit(font.render(f"Guess: {output.index(max(output))}", True, (255, 255, 255)), (0, 16))
        
    window.blit(outputSurface, position)
        
digit = mnistDigit((16, 16))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
            
    mouseButtons = pygame.mouse.get_pressed()
    if mouseButtons[0]:
        digit.draw(pygame.mouse.get_pos(), 1, 1.5)
        mason.generate_output(digit.contents)
    elif mouseButtons[2]:
        digit.erase(pygame.mouse.get_pos(), 2, 2.5)
        mason.generate_output(digit.contents)
        
    
                    
    window.fill(backgroundColour)
    
    digit.render_input()
    display_mason_output((0, 256))
    display_mason_activations((255, 36))
    
    window.blit(font.render("MASON number", True, (255, 255, 255)), (255, 0))
    window.blit(font.render("recognition test!!!", True, (255, 255, 255)), (255, 16))
    
    pygame.display.update()