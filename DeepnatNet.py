import numpy as np

import tensorflow as tf

class DeepnatNet():
    def __init__(self,learning_rate=0.0001):
    
        self.learning_rate = learning_rate
        
        self.__build()
    
    def __build(self):
        self.__init_global_step()
        self.__init_global_epoch()
        self.__init_input()
        
        x = _conv_block(self.input_image,'conv_block_1',32,7,1,'valid',dropout=0.5,is_training=self.is_training)#none,17,17,17,32
        x = _conv_block(x,'conv_block_2',32,3,2,'same',dropout=0.5,is_training=self.is_training)#none,9,9,9,32
        x = _conv_block(x,'conv_block_3',64,5,1,'valid',dropout=0.5,is_training=self.is_training)#none,5,5,5,64
        x = _conv_block(x,'conv_block_4',64,3,1,'valid',dropout=0.5,is_training=self.is_training)#none,3,3,3,64
        x = _conv_block(x,'conv_block_5',1024,3,1,'valid',dropout=0.5,is_training=self.is_training)#none,1,1,1,1024
        x = _conv_block(x,'conv_block_6',512,1,1,'valid',dropout=0.5,is_training=self.is_training)#none,1,1,1,512
        
        self.pred_list = []
        for i in range(27):
            pred = tf.reshape(
                              _conv_block(x,'pred_{}'.format(i+1),2+25,1,1,'valid',norm=None,activate=None,dropout=0,is_training=self.is_training),
                              tf.stack([tf.shape(x)[0],-1])
                             )#none,2+25
            #pred_forward_ground = tf.sigmoid(pred[:,:2])#none,2
            #pred_structure = tf.sigmoid( tf.expand_dims(pred_forward_ground[:,1],axis=1)*pred[:,2:] )#none,25
            pred_forward_ground = pred[:,:2]
            pred_structure = tf.expand_dims(pred_forward_ground[:,1],axis=1)*pred[:,2:]
            
            self.pred_list.append( tf.concat([pred_forward_ground[:,:2],pred_structure],axis=1) )#[(none,2+25),...]
        
        self.__init_output()
        
    def __init_output(self):
        with tf.variable_scope('output'):
            self.loss = 0
            for i,pred in enumerate(self.pred_list):
                self.loss += self.__loss(pred,self.label[:,i])
            self.loss = tf.reduce_mean(self.loss)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss)
            
            for i,pred in enumerate(self.pred_list):
                self.forward_ground_accuracy = tf.reduce_mean(
                                                              tf.cast(
                                                                      tf.equal(
                                                                                tf.cast(self.label[:,i]>0,tf.int32), 
                                                                                tf.argmax(pred[:,:2], axis=-1, output_type=tf.int32)
                                                                               ),
                                                                      tf.float32)
                                                             )
                self.structure_accuracy = tf.reduce_mean(
                                                          tf.cast(
                                                                  tf.equal(
                                                                            self.label[:,i], 
                                                                            tf.argmax(tf.concat([tf.expand_dims(pred[:,0],axis=1),tf.expand_dims(pred[:,1],axis=1)*pred[:,2:]],axis=-1), axis=-1, output_type=tf.int32)
                                                                           ),
                                                                  tf.float32)
                                                         )
        
        with tf.name_scope('train-summary-per-iteration'):
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('forward_ground_accuracy',self.forward_ground_accuracy)
            tf.summary.scalar('structure_accuracy',self.structure_accuracy)
            self.summaries_merged = tf.summary.merge_all()
    def __loss(self,pred,label):
        #label_forward_ground = tf.cast(label>0,tf.int32)#none,
        #loss_forward_ground = tf.reduce_sum(tf.square(pred[:,:2]-tf.one_hot(label_forward_ground,2)),axis=-1)
        #loss_structure = tf.reduce_sum(tf.square(tf.concat([tf.expand_dims(pred[:,0],axis=1),pred[:,2:]],axis=-1)-tf.one_hot(label,26)),axis=-1)
        loss_forward_ground = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.cast(label>0,tf.int32))
        loss_structure = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.concat([tf.expand_dims(pred[:,0],axis=1),pred[:,2:]],axis=-1), labels=label)
        return loss_forward_ground+loss_structure
    def __init_input(self):
        with tf.variable_scope('input'):
            self.input_image = tf.placeholder(tf.float32,[None,23,23,23,1],name='zsc_input')#训练、测试用
            self.is_training = tf.placeholder(tf.bool,name='zsc_is_train')#训练、测试用
            self.dropout = tf.placeholder(tf.float32,name='zsc_dropout')#训练、测试用
            self.label = tf.placeholder(tf.int32,[None,27])#训练用
    def __init_global_epoch(self):
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)
    def __init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

################################################################################################################
################################################################################################################
################################################################################################################
###LAYER
##group_norm
def _max_divisible(input,max=1):
    for i in range(1,max+1)[::-1]:
        if input%i==0:
            return i
def group_norm(x, eps=1e-5, name='group_norm') :
    with tf.variable_scope(name):
        _, _, _, _, C = x.get_shape().as_list()
        G = _max_divisible(C,max=C//2+1)
        G = min(G, C)
        if C%32==0:
            G = min(G,32)
        
        x = tf.reshape(x,tf.concat([tf.shape(x)[:4],tf.constant([G,C//G])],axis=0))#none,none,none,none,G,C//G
        
        mean, var = tf.nn.moments(x, [1, 2, 3, 4], keep_dims=True)#none,none,none,none,G,C//G
        x = (x - mean) / tf.sqrt(var + eps)#none,none,none,none,G,C//G
        
        x = tf.reshape(x,tf.concat([tf.shape(x)[:4],tf.constant([C])],axis=0))#none,none,none,none,C

        gamma = tf.Variable(tf.ones([C]), name='gamma')
        beta = tf.Variable(tf.zeros([C]), name='beta')
        gamma = tf.reshape(gamma, [1, 1, 1, 1, C])
        beta = tf.reshape(beta, [1, 1, 1, 1, C])

    return x* gamma + beta
##LeakyRelu
def LeakyRelu(x, leak=0.1, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)
##selu
def selu(x,name='selu'):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)
##_conv_block
def _conv_block(x,name,num_filters,kernel_size=3,stride=1,padding='same',norm='group_norm',activate='selu',dropout=0.5,is_training=True):
    use_norm = True if norm in ['batch_norm','group_norm'] else False
    with tf.variable_scope(name):
        x = tf.layers.conv3d(x,
                             filters=num_filters,
                             kernel_size=kernel_size,
                             strides=stride,
                             use_bias=False if use_norm else True,
                             padding=padding,
                             kernel_initializer=tf.glorot_uniform_initializer(),
                             name=name+'_conv')
        if use_norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name=name+'_batchnorm')
        elif use_norm=='group_norm':
            x = group_norm(x,name=name+'_groupnorm')
        else:
            pass
        if activate=='leaky':
            x = LeakyRelu(x,leak=0.1, name=name+'_leaky')
        elif activate=='selu':
            x = selu(x,name=name+'_selu')
        else:
            pass
        if dropout:
            x = tf.nn.dropout(x,dropout)
        else:
            pass
        
        return x
################################################################################################################
################################################################################################################
################################################################################################################
if __name__=='__main__':
    
    model = DeepnatNet()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(5):
            _,loss,acc1,acc2 = sess.run([model.train_op,model.loss,model.forward_ground_accuracy,model.structure_accuracy],
                           feed_dict={model.input_image:100*np.ones((3,23,23,23,1)),
                                      model.is_training:True,
                                      model.dropout:0.5,
                                      model.label:np.zeros((3,27))})
        
            print(loss,acc1,acc2)
    