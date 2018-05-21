import tensorflow as tf
from tensorflow.python.framework import graph_util

import numpy as np
import cv2

from DeepnatNet import DeepnatNet

from dataloader import BatchGenerator

import os
import glob

root = os.getcwd()

num_epochs = 100000
max_to_keep = 4
save_model_every = 1
test_every = 1

is_train = True

batch_size = 3

def main():
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)

    trainer = Train(sess)

    if is_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            trainer.save_model()
    else:
        print("Testing...")
        trainer.test()
        print("Testing Finished\n\n")

class Train:
    def __init__(self, sess):
        self.sess = sess

        self.train_data = BatchGenerator(
                                         DatasetPath=os.path.join(root,'processed_dataset'),batch_size=batch_size
                                        )
        self.test_data = BatchGenerator(
                                         DatasetPath=os.path.join(root,'processed_dataset'),batch_size=batch_size
                                        )
        
        print("Building the model...")
        self.model = DeepnatNet()
        print("Model is built successfully\n\n")
        tf.profiler.profile(tf.get_default_graph(),options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter(), cmd='scope')
        
        self.saver = tf.train.Saver(max_to_keep=max_to_keep,
                                    keep_checkpoint_every_n_hours=10,
                                    save_relative_paths=True)
        
        self.save_checkpoints_path = os.path.join(os.getcwd(),'checkpoints')
        if not os.path.exists(self.save_checkpoints_path):
            os.mkdir(self.save_checkpoints_path)

        self.init = None
        self.__init_model()

        self.__load_model()
        
        summary_dir = os.path.join(os.getcwd(),'logs')
        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)
        summary_dir_train = os.path.join(summary_dir,'train')
        if not os.path.exists(summary_dir_train):
            os.mkdir(summary_dir_train)
        summary_dir_test = os.path.join(summary_dir,'test')
        if not os.path.exists(summary_dir_test):
            os.mkdir(summary_dir_test)
        self.train_writer = tf.summary.FileWriter(summary_dir_train,sess.graph)
        self.test_writer = tf.summary.FileWriter(summary_dir_test)

    def __init_model(self):
        print("Initializing the model...")
        self.init = tf.group(tf.global_variables_initializer())
        self.sess.run(self.init)
        print("Model initialized\n\n")

    def save_model(self):
        print("Saving a checkpoint")
        self.saver.save(self.sess, self.save_checkpoints_path+'\\'+'yolov3', self.model.global_step_tensor)
        print("Checkpoint Saved\n\n")
        
    def __load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.save_checkpoints_path)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Checkpoint loaded\n\n")
        else:
            print("First time to train!\n\n")
    
    def train(self):
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, num_epochs + 1, 1):

            batch = 0
            
            loss_list = []
            forwardground_acc_list = []
            structure_acc_list = []
            
            for input_data,input_label in self.train_data.next():
                print('Training epoch:{},batch:{}'.format(cur_epoch,batch))
                
                cur_step = self.model.global_step_tensor.eval(self.sess)
                
                feed_dict={self.model.input_image:100*np.ones((3,23,23,23,1)),
                           self.model.is_training:True,
                           self.model.dropout:0.5,
                           self.model.label:np.zeros((3,27))}
                
                _, loss, summaries_merged,forward_ground_accuracy,structure_accuracy = self.sess.run(
                    [self.model.train_op, self.model.loss, self.model.summaries_merged,self.model.forward_ground_accuracy,self.model.structure_accuracy],
                    feed_dict=feed_dict)
                    
                print('batch-'+str(batch)+'|'+'loss:'+str(loss)+'|'+'forwardground_acc:'+str(forward_ground_accuracy)+'|'+'structure_acc:'+str(structure_accuracy)+'\n')
                loss_list += [loss]
                forwardground_acc_list += [forward_ground_accuracy]
                structure_acc_list += [structure_accuracy]

                self.model.global_step_assign_op.eval(session=self.sess,
                                                      feed_dict={self.model.global_step_input: cur_step + 1})

                self.train_writer.add_summary(summaries_merged,cur_step)

                if batch*batch_size > self.train_data.__len__():
                    batch = 0
                
                    avg_loss = np.mean(loss_list).astype(np.float32)
                    avg_forwardground_acc = np.mean(forwardground_acc_list).astype(np.float32)
                    avg_structure_acc = np.mean(structure_acc_list).astype(np.float32)
                    
                    self.model.global_epoch_assign_op.eval(session=self.sess,
                                                           feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                    print('Epoch-'+str(cur_epoch)+'|'+'avg loss:'+str(avg_loss)+'|'+'avg forwardground_acc:'+str(avg_forwardground_acc)+'|'+'avg structure_acc:'+str(avg_structure_acc)+'\n')
                    break
                
                if batch==0 and cur_epoch%99==0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    
                    _,summaries_merged = self.sess.run([self.model.train_op, self.model.summaries_merged],
                                                   feed_dict=feed_dict,
                                                   options=run_options,
                                                   run_metadata=run_metadata)
                    
                    self.train_writer.add_run_metadata(run_metadata, 'epoch{}batch{}'.format(cur_epoch,cur_step))
                    self.train_writer.add_summary(summaries_merged, cur_step)

                batch += 1
            
            if cur_epoch % save_model_every == 0 and cur_epoch != 0:
                self.save_model()
            
            if cur_epoch % test_every == 0:
                print('start test')
                #self.test()
                print('end test')
    def test(self):
        labels = ['raccoon']
        if not os.path.exists(os.path.join(os.getcwd(),'test_results')):
            os.mkdir(os.path.join(os.getcwd(),'test_results'))
        
        for image_path in glob.glob(os.path.join(os.getcwd(),'test_images','*.jpg')):
            image_name = image_path.split('\\')[-1]
            print('processing image {}'.format(image_name))

            image = cv2.imread(image_path)
            image_h,image_w,_ = image.shape
            _image = cv2.resize(image,(32*9,32*9))[np.newaxis,:,:,::-1]
            infos = self.sess.run(self.model.infos,feed_dict={self.model.input_image:_image,self.model.original_wh:[[image_w,image_h]]})
            print(infos)
            image = self.draw_boxes(image, infos.tolist(), labels)
            cv2.imwrite(os.path.join(os.getcwd(),'test_results',image_name),image)

if __name__=='__main__':
    main()