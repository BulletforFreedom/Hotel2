import tensorflow as tf
import datetime
import time

class DenseNet40:
  def __init__(self,n_classes=2):#,trn_img_list,trn_lbl_list,tst_img_list,tst_lbl_list
    self.sess=tf.Session()
    self.weight_decay=1e-4
    self.capacity=256
    self.n_classes=n_classes
    self.image_dim = 640     #size of resized image
    self.learning_rate=0.0001
    self.layers = 7
    self.growth=12 #growth rate=12
    self.max_step=10000
    with tf.name_scope('inputs'):
      self.xs = tf.placeholder(tf.float32, shape=[None, self.image_dim, self.image_dim, 3], name='x_input')
      self.ys= tf.placeholder(tf.int64, shape=[None], name='y_input')
    self.keep_prob = tf.placeholder(tf.float32)
    self.is_training = tf.placeholder("bool", shape=[])
    self.coord = tf.train.Coordinator()
    self.f=open('aaaa', 'a')


  def __destroy__(self):
    self.sess.close() 
    
  def get_batch(self,image_list,label_list,batch_size):
    #make images into batchsize
    if(batch_size==1):
      num_threads=1
    else:
      num_threads=32
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int64)
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_images(image, [self.image_dim, self.image_dim], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image) 
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=num_threads, capacity=self.capacity)  
    return image_batch, label_batch

  def weight_variable(self,shape):
    with tf.name_scope('weight'):
      initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    with tf.name_scope('bias'):
      initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

  def conv2d(self,input, in_features, out_features, kernel_size, strides, with_bias=False): 
    W = self.weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    with tf.name_scope('conv2d'):
      conv = tf.nn.conv2d(input, W, [ 1, strides, strides, 1 ], padding='SAME')
    if with_bias:
      return conv + self.bias_variable([ out_features ])
    return conv

  def batch_activ_conv(self,current, in_features, out_features, kernel_size, strides):
    with tf.name_scope('BN_ReLU_Conv'):
      current = tf.contrib.layers.batch_norm(current, scale=True, is_training=self.is_training, updates_collections=None)
      current = tf.nn.relu(current)
      current = self.conv2d(current, in_features, out_features, kernel_size,strides) 
      with tf.name_scope('dropout'):
        current = tf.nn.dropout(current, self.keep_prob)
    return current

  def block(self,input, in_features):
    current = input
    features = in_features
    for idx in range(self.layers):
      #tmp = batch_activ_conv(current, features, 4*self.growth, 1)   #bottleneck layer        
      tmp = self.batch_activ_conv(current, features, self.growth, 3, 1) #ksize=3 strides=1
      current = tf.concat((current, tmp), axis=3)
      features += self.growth
    return current, features

  def transition_layers(self,current, in_features, out_features, ksize, strides):
    current = self.batch_activ_conv(current, in_features, out_features, ksize, strides) 
    current = self.avg_pool(current, 2)#ksize=2 strides=2
    return current
    
  def avg_pool(self,input, strides):
    with tf.name_scope('avg_pool'):
      avp=tf.nn.avg_pool(input, [ 1, strides, strides, 1 ], [1, strides, strides, 1 ], 'VALID')
    return avp

  def network(self):

    current = self.conv2d(self.xs, 3, 16, 11, 3) # ksize=11 srides=3
    #---Dense block1---#
    with tf.name_scope('Denseblock1'):
      current, features = self.block(current,  16)
    #---Transition layers---#
    with tf.name_scope('transition_layers1'):
      current = self.transition_layers(current, features, 48, 1, 1) #output 48 ksize=1 srides=1
    #---Dense block2---#
    with tf.name_scope('Denseblock2'):
      current, features = self.block(current,  48)
    #---Transition layers---#
    with tf.name_scope('transition_layers2'):
      current = self.transition_layers(current, features, 96, 1, 1) #output 96 ksize=1 srides=1
    #---Dense block3---#
    with tf.name_scope('Denseblock3'):
      current, features = self.block(current,  96)
    #---Transition layers---#
    with tf.name_scope('transition_layers3'):
      current = self.transition_layers(current, features, 150, 1, 1) #output 150 ksize=1 srides=1
    #---Dense block4---#
    with tf.name_scope('Denseblock4'):
      current, features = self.block(current,  150)
    #---Transition layers---#
    with tf.name_scope('transition_layers4'):
      current = self.transition_layers(current, features, 192, 1, 1) #output 192 ksize=1 srides=1
    #---Dense block5---#
    with tf.name_scope('Denseblock5'):
      current, features = self.block(current,  192)
    #---global average pooling---#
    with tf.name_scope('global_average_pooling'):
      #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=self.is_training, updates_collections=None)
      current = tf.nn.relu(current)
      last_pool_kernel = int(current.get_shape()[-2])
      current = self.avg_pool(current, last_pool_kernel)

    current = tf.reshape(current, [ -1, features ]) #faltten

    with tf.name_scope('FC'):
      Wfc = self.weight_variable([ features, self.n_classes ])
      bfc = self.bias_variable([ self.n_classes ])
      with tf.name_scope('activation'):
        ys_ = tf.add(tf.matmul(current, Wfc), bfc)
    return ys_
    
  def cross_entropy(self,ys_):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ys_, labels=self.ys))
    
  def optimizer(self,cross_entropy):
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    with tf.name_scope('train_step'):
      train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy + l2 * self.weight_decay)
    return train_step
    
  def accuracy(self,ys_):
    correct_prediction = tf.equal(tf.argmax(ys_,1), self.ys)
    test_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return test_acc
    
  def trainNN(self,image_batch1, label_batch1,image_batch2, label_batch2,save_dir):
    
    ys_=self.network()
    cross_entropy=self.cross_entropy(ys_)
    train_step=self.optimizer(cross_entropy)
    test_acc=self.accuracy(ys_)
    
    self.sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver(max_to_keep=1)

    threads = tf.train.start_queue_runners(sess=self.sess,coord=self.coord)

    max_acc=0.0
    t_acc=0.0
    count_acc=0

    try:
      for step in range(self.max_step):
        if self.coord.should_stop() :
          break
        trn_images,trn_labels = self.sess.run([image_batch1, label_batch1])
        _,loss=self.sess.run([train_step,cross_entropy], feed_dict={self.xs:trn_images, self.ys:trn_labels, self.is_training: True,self.keep_prob:0.5})
        print(loss.shape)
        if ((step+1)%25)==0:#each 25 steps, print losss
           print("step: %d, train loss: %.2f" %(step+1,loss))

        if ((step+1)%50)==0:#each 50 steps, print accuracy
           test_images,test_labels = self.sess.run([image_batch2, label_batch2])
           t_acc=self.sess.run(test_acc, feed_dict={self.xs:test_images, self.ys:test_labels, self.is_training: False,self.keep_prob:1.})
           print("step: %d, test accuracy: %.2f" %(step+1, t_acc))

        if max_acc!=1. and (step+1)>=(self.max_step*0.7) and t_acc>=max_acc: #when steps > 70% max_step,save the model of the best result
           max_acc=t_acc
           saver.save(self.sess,save_dir,global_step=step+1)

        if t_acc==1.0 :
           count_acc+=1
           max_acc=t_acc
           saver.save(self.sess,save_dir,global_step=step+1)

        if count_acc==5:#if this model has 5th time with 100% accuracy rate, then break out of iteration
           break
    except tf.errors.OutOfRangeError:
        print("done!")
    finally:
      self.coord.request_stop()
    self.coord.join(threads)
    
  def testNN(self,tst_img_list,image_batch2, label_batch2,restore_dir):
    
    numpic=len(tst_img_list)
    
    ys_=self.network()
    test_acc=self.accuracy(ys_)

    self.sess.run(tf.global_variables_initializer())

    threads = tf.train.start_queue_runners(sess=self.sess,coord=self.coord)

    #count_img=0
    count_acc=0.0

    saver=tf.train.Saver()
    model_file=tf.train.latest_checkpoint(restore_dir)
    saver.restore(self.sess,model_file)    
    date=datetime.datetime.now().strftime('%Y-%m-%d')
    self.f.write(date+'\n') 

    try:
      for step in range(numpic):#numpic
        if self.coord.should_stop() :
          break
        t1=time.time()
        test_images,test_labels = self.sess.run([image_batch2, label_batch2])        
        t_acc=self.sess.run(test_acc, feed_dict={self.xs:test_images, self.ys:test_labels, self.is_training: False,self.keep_prob:1.})
        if(t_acc==1):
          count_acc+=1
          print("step: %d, recognised successfully"%(step+1))
        else:
          print("step: %d, recognised unsuccessfully"%(step+1))
          now_time = datetime.datetime.now().strftime('%H:%M:%S')
          self.f.write(now_time+' '+ tst_img_list[step]+'\n') 
        t2=time.time()
        print("Read image time %.2f"%(t2-t1))
      acc_rate=count_acc/numpic
      print("Accuracy rate: %.2f"%(acc_rate))
    except tf.errors.OutOfRangeError:
      print("done!")
    finally:
      self.coord.request_stop()
    self.coord.join(threads)
    
  def graph(self):
    ys_=self.network()
    with tf.name_scope('cross_entropy'):
      cross_entropy=self.cross_entropy(ys_)
    with tf.name_scope('optimizer'):
      train_step=self.optimizer(cross_entropy)
    writer=tf.summary.FileWriter("logs/",self.sess.graph)
    self.sess.run(tf.global_variables_initializer())
