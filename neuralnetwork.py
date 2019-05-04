import numpy as np

def sigmoid(x): #sigmoid function of vector x 
	return 1/(1+np.exp(-x))

def sigmoid_derivative(x): 
	return x*(1-x)

class NeuralNetwork():
	
	def __init__(self,x,y,layers_size):
		
		self.training_inputs = x #training input
		self.input_size = len(x[0])
		
		self.training_outputs = y #training_output
		self.output_size = 1
		
		self.layers_size = layers_size #encodes the size of each layer
		self.n_layers = len(self.layers_size) #number of layers
		
		self.weights=[np.random.rand(self.input_size,layers_size[0])]+[np.random.rand(self.layers_size[i],self.layers_size[i+1]) for i in range(self.n_layers-1)]+[np.random.rand(self.layers_size[-1],self.output_size)] #initialize random weights
		
	#Passes inputs through the neural network to get output
	def feedforward(self,inputs):
		layers = inputs
		for i in range(self.n_layers+1):
			layers = sigmoid(np.dot(layers,self.weights[i]))
		return layers
	
	#Training of the model by adjusting synaptic weight 	
	def backpropagate(self):
		#Pass training set through neural network
		output = self.feedforward(self.training_inputs)
		#Calculate error
		error = training_outputs - output
		
