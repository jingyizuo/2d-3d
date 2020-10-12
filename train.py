from __future__ import print_function
 
import argparse
from random import shuffle
import random
import os
import sys
import math
import tensorflow.compat.v1 as tf
import glob
import cv2
 
from image_reader import *
from net import *
 
parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--snapshot_dir", default="./snapshots", help="path of snapshots") 
parser.add_argument("--out_dir", default="./train_out", help="path of train outputs") 
parser.add_argument("--image_size", type=int, default=256, help="load image size") 
parser.add_argument("--random_seed", type=int, default=1234, help="random seed")
parser.add_argument('--base_lr', type=float, default=0.00001, help='initial learning rate for adam')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch') 
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument("--summary_pred_every", type=int, default=200, help="times to summary.") 
parser.add_argument("--write_pred_every", type=int, default=1000, help="times to write.") 
parser.add_argument("--save_pred_every", type=int, default=5000, help="times to save.") 
parser.add_argument("--lamda_l1_weight", type=float, default=100.0, help="L1 lamda") 
parser.add_argument("--lamda_gan_weight", type=float, default=1.0, help="GAN lamda") 
parser.add_argument("--train_picture_format", default=".tif", help="format of training datas.") 
parser.add_argument("--train_label_format", default=".tif", help="format of training labels.") 
parser.add_argument("--train_picture_path", default="./input/train/", help="path of training datas.") 
parser.add_argument("--train_label_path", default="./label/train/", help="path of training labels.") 
 
args = parser.parse_args() 
EPS = 1e-12 
#save models
def save(saver, sess, logdir, step): 
   model_name = 'model' 
   checkpoint_path = os.path.join(logdir, model_name)
   if not os.path.exists(logdir): 
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')
 
def cv_inv_proc(img):
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32) 
 
def get_write_picture(picture, gen_label, label, height, width):
    picture_image = cv_inv_proc(picture)
    gen_label_image = cv_inv_proc(gen_label[0]) 
    label_image = cv_inv_proc(label) 
    inv_picture_image = cv2.resize(picture_image, (width, height)) 
    inv_gen_label_image = cv2.resize(gen_label_image, (width, height)) 
    inv_label_image = cv2.resize(label_image, (width, height))
    output = np.concatenate((inv_picture_image, inv_gen_label_image, inv_label_image), axis=1)
    return output
    
def l1_loss(src, dst): 
    return tf.reduce_mean(tf.abs(src - dst))
 
def main(): 
    if not os.path.exists(args.snapshot_dir): 
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): 
        os.makedirs(args.out_dir)
    train_picture_list = glob.glob(os.path.join(args.train_picture_path, "*")) 
    tf.set_random_seed(args.random_seed) 
    #tf.compat.v1.disable_eager_execution()
    train_picture = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 1],name='train_picture') 
    train_label = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 1],name='train_label') 
 
    gen_label = generator(image=train_picture, gf_dim=64, reuse=False, name='generator')
    dis_real = discriminator(image=train_picture, targets=train_label, df_dim=64, reuse=False, name="discriminator")
    dis_fake = discriminator(image=train_picture, targets=gen_label, df_dim=64, reuse=True, name="discriminator") 
 
    gen_loss_GAN = tf.reduce_mean(-tf.log(dis_fake + EPS)) 
    gen_loss_L1 = tf.reduce_mean(l1_loss(gen_label, train_label))
    gen_loss = gen_loss_GAN * args.lamda_gan_weight + gen_loss_L1 * args.lamda_l1_weight 
 
    dis_loss = tf.reduce_mean(-(tf.log(dis_real + EPS) + tf.log(1 - dis_fake + EPS))) 
 
    gen_loss_sum = tf.summary.scalar("gen_loss", gen_loss) 
    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss) 
 
    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph()) 
 
    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name] 
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name] 
 
    d_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1) 
    g_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1) 
 
    d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars) 
    d_train = d_optim.apply_gradients(d_grads_and_vars) 
    g_grads_and_vars = g_optim.compute_gradients(gen_loss, var_list=g_vars) 
    g_train = g_optim.apply_gradients(g_grads_and_vars) 
 
    train_op = tf.group(d_train, g_train) 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    sess = tf.Session(config=config) 
    init = tf.global_variables_initializer() 
 
    sess.run(init) 
 
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50) #模型保存器
 
    counter = 0 
 
    for epoch in range(args.epoch): 
        shuffle(train_picture_list) 
        for step in range(len(train_picture_list)): 
            counter += 1
            picture_name, _ = os.path.splitext(os.path.basename(train_picture_list[step])) 
            picture_resize, label_resize, picture_height, picture_width = ImageReader(file_name=picture_name, picture_path=args.train_picture_path, label_path=args.train_label_path, picture_format = args.train_picture_format, label_format = args.train_label_format, size = args.image_size)
            batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis = 0) 
            batch_label = np.expand_dims(np.array(label_resize).astype(np.float32), axis = 0) 
            feed_dict = { train_picture : batch_picture, train_label : batch_label } 
            gen_loss_value, dis_loss_value, _ = sess.run([gen_loss, dis_loss, train_op], feed_dict=feed_dict) 
            if counter % args.save_pred_every == 0:
                save(saver, sess, args.snapshot_dir, counter)
            if counter % args.summary_pred_every == 0: 
                gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss_sum, dis_loss_sum], feed_dict=feed_dict)
                summary_writer.add_summary(gen_loss_sum_value, counter)
                summary_writer.add_summary(discriminator_sum_value, counter)
            if counter % args.write_pred_every == 0: 
                gen_label_value = sess.run(gen_label, feed_dict=feed_dict)
                write_image = get_write_picture(picture_resize, gen_label_value, label_resize, picture_height, picture_width) 
                write_image_name = args.out_dir + "/out"+ str(counter) + ".png" 
                cv2.imwrite(write_image_name, write_image) 
            print('epoch {:d} step {:d} \t gen_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, gen_loss_value, dis_loss_value))
    
if __name__ == '__main__':
    main()
