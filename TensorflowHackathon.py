import tensorflow as tf
import numpy as np



BATCH_SIZE = 256
TRAINING_ITER = 1000000
LEARNING_RATE = 0.00008



class Dataset:
    
    def __init__(self, directory):
        self.directory = directory
        self.file = open(self.directory,"r")
        
        self.list = []
        
        for i, line in enumerate(self.file):
            if(line != "\n"):
                self.list.append(line)
        self.length = len(self.list)
        self.file.close()
    
    def getRandomInstance(self):
        random_num = np.random.randint(low=0,high=self.length-2,size=1)
        return self.list[int(random_num)]
    
    def getInstance(self,index):
        return self.list[index]
    
    @staticmethod
    def convertTxtToArray(line):

        line = line.split('),')
        line = [i[1:] for i in line]
        
        label = np.zeros(shape=(5),dtype=np.float32)
        data = np.zeros(shape=(7,5),dtype=np.float32)
        
        for i in range(0,len(line)):
        
            if i == (len(line)-1):
                
                label = np.fromstring(line[i][:len(line[i])-2],dtype=np.float32,sep=',')
                label[0] = label[0]/5.0
                
                
            else:
                data[i] = np.fromstring(line[i],dtype=np.float32,sep=',')
                data[i][0] = data[i][0]/5.0
        
        return (np.expand_dims(data,axis=2),label)



class Model:
    
    def __init__(self):
        tf.reset_default_graph()
        self.build()
        
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        self.session.run(init)
        
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
    
    def save(self):
        self.saver.save(self.session, './models/model.ckpt')
        print("saved trainable variables")
        
    def close(self):
        self.session.close()

    def getOutput(self,data):
        
        output = self.session.run([self.output],feed_dict={self.input:data})
        
        return output
    
    def build(self):
        
       
        self.input = tf.placeholder(dtype=tf.float32,shape=(None,7,5,1))
        self.label = tf.placeholder(dtype=tf.float32,shape=(None,5))

        #horizontal convolution filter
        hor_conv1 = tf.keras.layers.Conv2D(filters=128,kernel_size=(2,5),padding='VALID',activation='relu', use_bias=True)(self.input)
        print(hor_conv1.shape)
        hor_flatten = tf.reshape(hor_conv1,[-1,768])
        hor_fully_connected = tf.keras.layers.Dense(use_bias=True,activation='relu',units=768)(hor_flatten)
        
        #vertical convolution filter
        ver_conv1 = tf.keras.layers.Conv2D(filters=128,kernel_size=(7,1),padding='VALID',activation='relu', use_bias=True)(self.input)
        print(ver_conv1.shape)
        ver_flatten = tf.reshape(ver_conv1,[-1,640])
        ver_fully_connected = tf.keras.layers.Dense(use_bias=True,activation='relu',units=640)(ver_flatten)
        
        fc_combined = tf.concat([hor_fully_connected,ver_fully_connected],axis=1)

        fc_combined_2 = tf.keras.layers.Dense(units=512,activation='relu',use_bias=True)(fc_combined)
     
        fc_combined_3 = tf.keras.layers.Dense(units=256,activation='relu',use_bias=True)(fc_combined_2)
        
        self.output = tf.keras.layers.Dense(units=5)(fc_combined_3)
       
        self.loss = tf.losses.mean_squared_error(labels=self.label,predictions=self.output)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        
        self.minimize = optimizer.minimize(loss=self.loss)
    
    def restore(self):
        self.saver.restore(self.session,"./models/model.ckpt")
    
    def train(self,data,label):
        
        output,loss,_ = self.session.run((self.output,self.loss,self.minimize),feed_dict={self.input:data,self.label:label})
        return loss
        

dataset = Dataset("C:\\Users\\Steven\\Desktop\\training.txt")


model = Model()
model.restore()

for i in range(0,TRAINING_ITER):
    
    batch_data = np.zeros(shape=(BATCH_SIZE,7,5,1),dtype=np.float32)
    batch_label = np.zeros(shape=(BATCH_SIZE,5),dtype=np.float32)
    
    for j in range(0,BATCH_SIZE):
        (data,label) = Dataset.convertTxtToArray(dataset.getRandomInstance())
        batch_data[j] = data
        batch_label[j] = label
    
    loss = model.train(batch_data,batch_label)
    
    if i%4000 == 0:
        model.save()
    if i%500 == 0:   
        print(loss)

model.close()



