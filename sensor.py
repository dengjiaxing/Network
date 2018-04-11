#coding:utf8
class Perception(object):
	"""docstring for Perceptron"""
	def __init__(self, input_num,activator):
		self.activator=activator
		self.weights=[0.0 for _ in range(input_num)]  #权重向量，初始化为0,根据input_num大小，初始化权重数组
		self.bias=0.0
		
	def __str_(self):
		return 'weights\t:%s\nbias\t:%f\n'%(self.weights,self.bias)    #print 

	def predict(self,input_vec):
		#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
		#zip([1,2,3],[4,5,6])=>[(1,4),(2,5),(3,6)]
		#mapp函数计算[x1*w1, x2*w2, x3*w3]
		#reduce求和
		return self.activator(reduce(lambda a,b:a+b,map(lambda (x,w):x*w,zip(input_vec,self.weights)),0.0)+self.bias)

	def train(self,input_vecs,labels,iteration,rate):

		#输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
		for i in range(iteration):
			self._one_iteration(input_vecs,labels,rate)

	def _one_iteration(self,input_vecs,labels,rate):

		#一次迭代，把所有的训练数据过一遍
		samples=zip(input_vecs,labels)
		for(input_vec,label) in samples:
			output=self.predict(input_vec)
			self._update_weight(input_vec,output,label,rate)

	def _update_weight(self,input_vec,output,label,rate):

		delta=label-output

		#然后利用感知器规则更新权重
		self.weights=map(lambda (x,w):w+rate*delta*x,zip(input_vec,self.weights))
		#更新bias
		self.bias+=rate*delta

		
def f(x):
	#定义激活函数f
	return 1 if  x>0 else 0

def get_training_dataset():

	input_vecs=[(1,1),[0,0],[1,0],[0,1]]
	labels=[1,0,0,0]
	return input_vecs,labels

def train_and_perception():
	#创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
	p=Perception(2,f)
	#训练，迭代10轮, 学习速率为0.1
	input_vecs,labels=get_training_dataset()
	p.train(input_vecs,labels,10,0.1)
	#返回训练好的感知器
	return p

if __name__ == '__main__':
	and_perception=train_and_perception()
	print and_perception

	print "1 and 1=%d"%(and_perception.predict([1,1]))
	print "0 and 0=%d"%(and_perception.predict([0,0]))
	print "1 and 0=%d"%(and_perception.predict([1,0]))
	print "0 and 1=%d"%(and_perception.predict([0,1]))

