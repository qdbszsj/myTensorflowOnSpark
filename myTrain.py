'''
尝试模仿雅虎给的demo，把CNN的开源训练代码改成他们的形式，但是tf的东西我不太熟，很多细节不好改，暂时放弃了，这份代码尚未完成

'''


import argparse
import os
import time

import tensorflow as tf

import inception_resnet_v1
from utils import inputs, get_files_name


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf

import argparse
import numpy
from datetime import datetime

from tensorflowonspark import TFCluster

sc = SparkContext(conf=SparkConf().setAppName("parker_spark"))
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

parser = argparse.ArgumentParser()
'''
parser.add_argument("--batch_size", help="number of records per batch", type=int, default=100)
parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
parser.add_argument("--format", help="example format: (csv|pickle|tfr)", choices=["csv", "pickle", "tfr"], default="csv")
parser.add_argument("--images", help="HDFS path to MNIST images in parallelized format")
parser.add_argument("--labels", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument("--model", help="HDFS path to save/load model during train/inference", default="mnist_model")
parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument("--readers", help="number of reader/enqueue threads", type=int, default=1)
parser.add_argument("--steps", help="maximum number of steps", type=int, default=1000)
parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("--mode", help="train|inference", default="train")
parser.add_argument("--rdma", help="use rdma connection", default=False)
'''
parser.add_argument("--format", help="example format: (csv|pickle|tfr)", choices=["csv", "pickle", "tfr"], default="csv")
parser.add_argument("--mode", help="train|inference", default="train")
parser.add_argument("--steps", help="maximum number of steps", type=int, default=1000)
parser.add_argument("--rdma", help="use rdma connection", default=False)

parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="Init learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Set 0 to disable weight decay")
parser.add_argument("--model_path", type=str, default="./models", help="Path to save models")
parser.add_argument("--log_path", type=str, default="./train_log", help="Path to save logs")
parser.add_argument("--epoch", type=int, default=6, help="Epoch")
parser.add_argument("--images", type=str, default="./data/train", help="Path of tfrecords")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--keep_prob", type=float, default=0.8, help="Used by dropout")
parser.add_argument("--cuda", default=False, action="store_true", help="Set this flag will use cuda when testing.")

args = parser.parse_args()
print("args:", args)

print("{0} ===== Start".format(datetime.now().isoformat()))

if args.format == "tfr":#tfrecord格式的文件，还要特殊处理一下
    images = sc.newAPIHadoopFile(args.images, "org.tensorflow.hadoop.io.TFRecordFileInputFormat", keyClass="org.apache.hadoop.io.BytesWritable", valueClass="org.apache.hadoop.io.NullWritable")
    def toNumpy(bytestr):
        example = tf.train.Example()
        example.ParseFromString(bytestr)
        features = example.features.feature
        image = numpy.array(features['image_raw'].int64_list.value)
        label = numpy.array(features['age'].int64_list.value)
	'''
		'age': _int64_feature(int(ages[index])),
                'gender': _int64_feature(int(genders[index])),
                'image_raw': _bytes_feature(image_raw),
                'file_name': _bytes_feature(str(file_name[index][0]))
	'''
        return (image, label)

    dataRDD = images.map(lambda x: toNumpy(bytes(x[0])))
							   

def main_fun(args, ctx):#这个函数是最核心的函数，会分到每个机器上执行
    from datetime import datetime
    import math
    import numpy
    import tensorflow as tf
    import time
	
    import inception_resnet_v1 from utils import inputs, get_files_name
    
    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index

	# Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

	#parameters
    batch_size = args.batch_size
	
	# Get TF cluster and server instances
    cluster, server = ctx.start_cluster_server(1, args.rdma)
    
    def feed_dict(batch):#这个函数要写在main_fun里面
        # Convert from [(images, labels)] to two numpy arrays of the proper type
        images = []
        labels = []
        for item in batch:
            images.append(item[0])
            labels.append(item[1])
        xs = numpy.array(images)
        xs = xs.astype(numpy.float32)
        xs = xs / 255.0
        ys = numpy.array(labels)
        ys = ys.astype(numpy.uint8)
        return (xs, ys)
	
    if job_name == "ps":#ps就是master
        server.join()
    elif job_name == "worker":	
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
	    images, age_labels, gender_labels, _ = inputs(path=get_files_name(image_path), batch_size=batch_size,
                                                      num_epochs=epoch)
				'''
				这里应该有一个placeholder放X和Y，也就是image和label
				结果这个dalao就放了一个bool类型，下文训练的时候还恒置为0了
				这里不知道这个inputs是个什么鬼，参数里还有batch_size，
				难道是每次运行的时候都会读取一个batch的数据？？？
				会不会这个images和labels本身就是placeholder，是另一种声明方法？
				'''
	    train_mode = tf.placeholder(tf.bool)
				
				
				'''
				这里就是CNN的核心函数，调用了inception_resnet_v1.inference，我不太懂这个原理，不知道这个
				images是什么类型的变量，是一个placeholder么
				
				另外，这些变量外面都应该套个name_scope，防止变量重名，这些东西之前都不懂。。。很容易改炸
				'''
	    age_logits, gender_logits, _ = inception_resnet_v1.inference(images, keep_probability=kp, phase_train=train_mode, weight_decay=wd)
																	 
																 
	    age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=age_labels, logits=age_logits)
            age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)
            gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gender_labels,
                                                                              logits=gender_logits)
            gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)

            # l2 regularization
            total_loss = tf.add_n([gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
            age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
            abs_loss = tf.losses.absolute_difference(age_labels, age)
            gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, gender_labels, 1), tf.float32))

            tf.summary.scalar("age_cross_entropy", age_cross_entropy_mean)
            tf.summary.scalar("gender_cross_entropy", gender_cross_entropy_mean)
            tf.summary.scalar("total loss", total_loss)
            tf.summary.scalar("train_abs_age_error", abs_loss)
            tf.summary.scalar("gender_accuracy", gender_acc)
		
            global_step = tf.train.get_or_create_global_step()
	    #global_step = tf.Variable(0, name="global_step", trainable=False)
            lr = tf.train.exponential_decay(start_lr, global_step=global_step, decay_steps=3000, decay_rate=0.9,
                                        staircase=True)
            optimizer = tf.train.AdamOptimizer(lr)
            tf.summary.scalar("lr", lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
			
	    with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(total_loss, global_step)
					
				
				# if you want to transfer weight from another model,please comment below codes
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
          	# if you want to transfer weight from another model, please comment above codes

				
				'''
				这上面的应该都没什么大问题，就是求损失函数，写optimizer，照着抄就行
				下面这有个writer，要写文件，我们分布式的怎么写？？？
				'''
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        		# if you want to transfer weight from another model,please uncomment below codes
        		# sess, new_saver = save_to_target(sess,target_path='./models/new/',max_to_keep=100)
        		# if you want to transfer weight from another model, please uncomment above codes

        		# if you want to transfer weight from another model,please comment below codes
            new_saver = tf.train.Saver(max_to_keep=100)
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
            	new_saver.restore(sess, ckpt.model_checkpoint_path)
            	print("restore and continue training!")
            else:
            	pass
        		# if you want to transfer weight from another model, please comment above codes
				
				
	    '''
		这里的ctx要注意了，是分布式特有的，应该是设置模型保存的绝对路径
		'''
        logdir = ctx.absolute_path(args.model)
	print("tensorflow model path: {0}".format(logdir))
		'''
		这里设置一下最大训练次数
		'''
	hooks = [tf.train.StopAtStepHook(last_step=100000)]
				
	if job_name == "worker" and task_index == 0:
            summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
				
	    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(task_index == 0), checkpoint_dir=logdir, hooks=hooks) as mon_sess:	
		step = 0
			'''
			这里又有ctx，是分布式特有的
			'''
                tf_feed = ctx.get_data_feed(args.mode == "train")
		while not mon_sess.should_stop() and not tf_feed.should_stop() and step < args.steps:
			    '''
		        下文讲道理，应该开始feed和训练，这里用feed_dict函数，我们自己写的
		        '''
		    batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
		    feed = {x: batch_xs, y_: batch_ys}
					
		    if len(batch_xs) > 0:
		        if args.mode == "train":
			    _, summary, step = mon_sess.run([train_op, merged, global_step], feed_dict=feed)
				        
			    if task_index == 0:
                                summary_writer.add_summary(summary, step)
				
	            if mon_sess.should_stop() or step >= args.steps:
                        tf_feed.terminate()	
					
    if job_name == "worker" and task_index == 0:
        summary_writer.close()
			
			
			
cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK, log_dir=args.model)
if args.mode == "train":
    cluster.train(dataRDD, args.epochs)
  
cluster.shutdown()  							   							  
		
'''
上面还没写怎么保存模型
原本他是这么写的，mnist里我也没看到在哪保存的模型
save_path = new_saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=global_step)
print("Model saved in file: %s" % save_path)
原文存的sess，我们这里得用mon_sess不知道会不会出什么幺蛾子
综上所述，坑很多，不好搞
'''

