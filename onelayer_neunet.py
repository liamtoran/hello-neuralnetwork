import numpy as np

class onelayer_neunet():
	def __init__(self):
		np.random.seed(1)
	def sigmoid(self,x): #sigmoid function of vector x 
		return 1/(1+np.exp(-x))
		
	def sigmoid_derivative(self,x): 
		return x*(1-x)
	#Training of the model by adjusting synaptic weight each time to get a better result	
	def train(self, training_inputs, training_outputs, training_iterations):
		for iteration in range(training_iterations):
			#Pass training set through neural network
			output = self.think(training_inputs)
			#Calculate error
			error = training_outputs - output
			#Adjust synaptic weights
			adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
			self.synaptic_weights += adjustments
	
	#Passes inputs through the neural network to get output
	def think(self,inputs):
		inputs = inputs.astype(float)
		output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
		return output


if __name__ ==  "__main__":
	# Initialize the single neuron neural network
	neural_network=onelayer_neunet()
	neural_network.synaptic_weights = 2 * np.random.random((3,1)) - 1
	print("Random synaptic weights: ")
	print(neural_network.synaptic_weights)
	
	#Training set, 4 exemples with 3 input values and 1 output value
	training_inputs =  np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_outputs = np.array([[0,1,1,0]]).T
	
	#Train the neural network
	neural_network.train(training_inputs,training_outputs,10000)
	print("Synaptic weights after training: ")
	print(neural_network.synaptic_weights)
	A  = str(input("1:"))
	B = str(input("2:"))
	C = str(input("3:"))
	print("New situation: input data =",A,B,C)
	print("Output data: ")
	print(neural_network.think(np.array([A,B,C])))
