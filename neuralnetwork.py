import numpy as np

def sigmoid(x): #sigmoid function of vector x 
	return 1/(1+np.exp(-x))

def sigmoid_derivative(x):  #Derivative of the sigmoid function
	return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork():
	
	def __init__(self,x,y,layers_size):
		
		self.training_inputs = x #training input
		self.input_size = x.shape[1] #each line of x is an input and each row corresponds to the initial neural activation
		self.batch_size = x.shape[0] #number of training inputs
		
		self.training_outputs = y #training_output
		self.output_size = y.shape[1] #each line of y is the ouput of the input and each row is the corresponding neural activation output. It is important that y is a matrix and never a vector.
		
		self.layers_size = layers_size #encodes the size of each layer
		self.n_layers = len(self.layers_size) #number of hidden layers
		
		
		"""For random initialization replace below with:
		self.weights=[np.random.randn(self.input_size,layers_size[0])]+[np.random.randn(self.layers_size[i],self.layers_size[i+1]) for i in range(self.n_layers-1)]+[np.random.randn(self.layers_size[-1],self.output_size)] """
		
		#initializes random weights with He et al initialization
		self.weights=[np.random.randn(self.input_size,layers_size[0])*np.sqrt(2/self.input_size)]
		self.weights+=[np.random.randn(self.layers_size[i],self.layers_size[i+1])*np.sqrt(2/self.layers_size[i]) for i in range(self.n_layers-1)]
		self.weights+=[np.random.randn(self.layers_size[-1],self.output_size)*np.sqrt(2/self.layers_size[-1])] 
		
		#Initialization of biases (zeros): 
		
		self.biases = [np.zeros(len(self.weights[i][0])) for i in range(self.n_layers+1)]
		
		
	#Passes inputs through the neural network to get output
	def feedforward(self,inputs):
		layers = [inputs] #activations
		z = [] #z vectors ( input of sigmoid)
		#passes previous layer through next step of the neural network
		for i in range(self.n_layers+1):
			z += [np.dot(layers[-1],self.weights[i])+self.biases[i]]
			layers += [sigmoid(z[-1])]
		return (z, layers)


	#Training of the model by adjusting synaptic weight via backpropagation	
	def backpropagate(self, z_s, a_s): 
		# y = training output, a_s =  neural activation of front propagation 
		#
		dw = [np.zeros(w.shape) for w in self.weights]
		db = [np.zeros(b.shape) for b in self.biases]
		deltas = [None] * len(self.weights)
		#Calculate final error
		deltas[-1] = (self.training_outputs-a_s[-1])*(sigmoid_derivative(z_s[-1]))
		#Perform BackPropagation
		#db[-1]=np.ones((self.batch_size,1)).transpose().dot(deltas)/self.batch_size
		#dw[-1]=np.dot(a_s[-2].transpose(),deltas)
		for i in range(2, self.n_layers+2):
			z = z_s[-i]
			sp = sigmoid_derivative(z)
			deltas[-i] = np.dot(deltas[-i+1], self.weights[-i+1].transpose())*sp
		db = [np.ones((self.batch_size,1)).transpose().dot(d)/self.batch_size for d in deltas]
		dw = [np.dot(a_s[i].transpose(), d)/self.batch_size for i,d in enumerate(deltas)]
		return (db,dw)
			

	
	def train(self,lr = 1):
		#Pass training set through neural network
		z_s, a_s = self.feedforward(self.training_inputs)
		db, dw = self.backpropagate(z_s,a_s)
		self.weights = [w+lr*dweight for w,dweight in zip(self.weights, dw)]
		self.biases = [b+lr*dbias for b,dbias in zip(self.biases, db)]
	
	def output(self,x):
		return self.feedforward(x)[1][-1]
		
		
		
	
		
if __name__ == "__main__":
	x = np.array([[0,0,1,1],[1,0,1,1],[1,1,0,1],[0,0,0,1],[1,1,1,1]])
	y = np.array([[0,0],[1,0],[1,1],[0,0],[1,1]])
	layers = np.array([3,2])
	nn= NeuralNetwork(x,y,layers)
	for i in range(10000):
		nn.train()
	print(nn.feedforward(x)[1][-1])
	#print(nn.train())
	#print(nn.errors)
