from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import namedtuple

from module import *
from utils import *
import utils

import cv2

class vae(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        #self.image_size = args.fine_size
        
        self.dataset_dir = args.dataset_dir

        self.feed_foward_network = feed_foward_network
        self.mae = mae_criterion
        self.mse = mse_criterion
        self.checkpoint_dir = args.pj_dir + 'checkpoint'
        self.logs_dir = args.pj_dir + 'logs'
        self.sample_dir = args.pj_dir + 'sample'
        self.test_dir =  args.pj_dir + 'test'


        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        self._build_model(args)
        
        self.saver = tf.train.Saver(max_to_keep=100)
        
        if args.phase == 'train':
            self.ds = pd.read_csv('../forward_model_data/dataset/KMAC_dataset_161steps/161_steps_dataset_nsc.csv')
            #self.ds = self.ds.iloc[:10]
        else:
            self.ds = pd.read_csv('../forward_model_data/dataset/KMAC_dataset_161steps/161_steps_dataset_nsc.csv')
            #self.ds = pd.read_csv('../forward_model_data/dataset/KMAC_new_RCWA_dataset_200626_nsc/KMAC_new_rcwa_data_preprocessed_nsc.csv')
        
    def _load_batch(self, dataset, idx):
        
        #filename_list = dataset.iloc[:,0][idx * self.batch_size:(idx + 1) * self.batch_size].values.tolist()

        # input batch (parameter)
        input_batch = dataset.iloc[:,:5][idx * self.batch_size:(idx + 1) * self.batch_size].values.tolist()

        # target batch (spectrum)
        target_batch = dataset.iloc[:,5:166][idx * self.batch_size:(idx + 1) * self.batch_size].values.tolist()

        return input_batch, target_batch #, filename_list


    def _build_model(self, args):
        # ref : https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py
        # log sigma ref : https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/


        self.input_param = tf.placeholder(tf.float32, [None, 5], name='input_l')
        self.spectrum_target = tf.placeholder(tf.float32, [None, 161], name='spectra_target')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # model
        self.spectra_pred = self.feed_foward_network(self.input_param, reuse=False, is_training=self.is_training)
        
        # loss
        self.loss = self.mse(self.spectra_pred, self.spectrum_target)

        self.loss_summary = tf.summary.scalar("loss", self.loss)

        self.t_vars = tf.trainable_variables()
        print("trainable variables : ")
        print(self.t_vars)
        

    def train(self, args):
        
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(self.lr, global_step, params['epoch_step'], 0.96, staircase=False)
        #learning_rate = tf.train.cosine_decay_restarts(self.lr, global_step,7500,m_mul=0.9)
        learning_rate = self.lr

        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, var_list=self.t_vars, global_step = global_step)
        self.optim = tf.group([train_op, update_ops])


        print("initialize")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        
        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(self.checkpoint_dir): 
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            
            batch_idxs = len(self.ds) // self.batch_size

            #ds_1 = self.ds
            ds_1 = self.ds.sample(frac=1)
            
            for idx in range(0, batch_idxs):

                input_batch, target_batch = self._load_batch(ds_1, idx)

                # Update network
                _, loss, c_lr, summary_str = self.sess.run([self.optim, self.loss, learning_rate, self.loss_summary],
                                                   feed_dict={self.input_param: input_batch, self.spectrum_target: target_batch, self.lr: args.lr, self.is_training: True})

                self.writer.add_summary(summary_str, counter)

                counter += 1
                if idx==batch_idxs-1:
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f loss: %4.7f lr: %4.7f" % (
                        epoch, idx, batch_idxs, time.time() - start_time, loss, c_lr)))

                if np.mod(counter, args.save_freq) == 20:
                    self.save(self.checkpoint_dir, counter)


    def save(self, checkpoint_dir, step):
        model_name = "dnn.model"
        model_dir = "%s" % (self.dataset_dir)
        #checkpoint_dir = checkpoint_dir + '/' + model_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        checkpoint_dir+'/'+model_name,
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        #model_dir = "%s" % (self.dataset_dir)
        #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt)
            ckpt_paths = ckpt.all_model_checkpoint_paths    #hcw
            print(ckpt_paths)
            ckpt_name = os.path.basename(ckpt_paths[-1])    #hcw # default [-1]
            #temp_ckpt = 'dnn.model-80520'
            #ckpt_name = os.path.basename(temp_ckpt)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0


        batch_idxs = len(self.ds) // self.batch_size

        #ds_1 = self.ds.sample(frac=1)
        ds_1 = self.ds
        #print(ds_1.iloc[:,4:][0:10].values.tolist())
        
        loss_list = []

        df_param_target_all = pd.DataFrame()
        df_param_pred_all = pd.DataFrame()

        for idx in range(0, batch_idxs):

            input_batch, target_batch = self._load_batch(ds_1, idx)

            #geo_pred, pred, loss, loss_r = self.sess.run([self.geo_reconstructed_l, self.spectra_l_predicted, self.total_loss, self.loss_r],
            #                                    feed_dict={self.input_param: input_batch, self.spectrum_target: target_batch})
            loss, pred = self.sess.run([self.loss, self.spectra_pred], feed_dict={self.input_param: input_batch, self.spectrum_target: target_batch})


            counter += 1
            if idx%1==0:
                print(("Step: [%4d/%4d] time: %4.4f" % (
                    idx, batch_idxs, time.time() - start_time)))
                #df_param = pd.DataFrame(np.squeeze(input_batch), columns={'param1','param2','param3','param4','param5'}) 
                df_pred = pd.DataFrame(np.squeeze(pred))
                df_target = pd.DataFrame(np.squeeze(target_batch))
                #df_geo_pred =  np.squeeze(geo_pred)

                #df_param_pred = pd.concat([df_param, df_pred], axis=1, sort=False)
                #df_param_target = pd.concat([df_param, df_target], axis=1, sort=False)
                #df_param_param = pd.concat([df_param, df_geo_pred], axis=1, sort=False)
                
                df_param_target_all = pd.concat([df_param_target_all, df_target], axis=0, sort=False)
                df_param_pred_all = pd.concat([df_param_pred_all, df_pred], axis=0, sort=False)


        df_param_target_all.to_csv(self.test_dir+'/result_test_target.csv', index=False)
        df_param_pred_all.to_csv(self.test_dir+'/result_test_prediction.csv', index=False)

            
        print("total time")
        print(time.time() - start_time)


    def test_reconstruction(self, args):

        self.batch_size = 1

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0



        batch_idxs = len(self.ds) // self.batch_size

        #ds_1 = self.ds.sample(frac=1)
        ds_1 = self.ds
        #print(ds_1.iloc[:,4:][0:10].values.tolist())
        
        loss_list = []

        for idx in range(0, batch_idxs):

            input_batch, target_batch, filename_list = self._load_batch(ds_1, idx)

            for j in range(5):
                latent_vector = list(np.random.normal(0,3,5))
                #for k in range(5):
                #    latent_vector[k] = j*0.5 - 2.5
                print(latent_vector)
                latent_vector = np.expand_dims(latent_vector, 0)
                geo_recon = self.sess.run([self.geo_reconstructed], 
                                            feed_dict={self.latent_vector: latent_vector, self.spectrum_target: target_batch})


                print(self.test_dir+'/reconstruction/')
                #print(np.shape(geo_pred))
                geo_recon = np.squeeze(geo_recon)
                cv2.imwrite(self.test_dir+'/'+str(filename_list)+'_'+str(latent_vector)+'.bmp',(geo_recon+1)*128)
            