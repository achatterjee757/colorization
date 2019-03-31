import tensorflow as tf
import cv2
import os
import numpy as np
from tqdm import tqdm

# image_path="flower_images"
image_path="flower_photos/All_Images"

image_data_gray=[]
image_data_color=[]
for file in tqdm(os.listdir(image_path)):
    try:
        color_image=cv2.imread(os.path.join(image_path,file))
        gray_scale_image=cv2.imread(os.path.join(image_path,file),cv2.IMREAD_GRAYSCALE)
        color_image=cv2.resize(color_image,(256,256))
        gray_scale_image=cv2.resize(gray_scale_image,(256,256))
        gray_scale_image=np.reshape(gray_scale_image,[256,256,1])
        image_data_gray.append(gray_scale_image)
        image_data_color.append(color_image)
    except:
        pass
# cv2.imshow("sfsdf",image_data_color[0])
# cv2.imshow("sfsdf12",image_data_gray[0])
# cv2.waitKey(0)

image_data_color=(np.array(image_data_color)/127)-1
image_data_gray=(np.array(image_data_gray)/127)-1

train_input_color=image_data_color[10:,:,:,:]
train_input_gray=image_data_gray[10:,:,:,:]
test_input_color=image_data_color[0:10,:,:,:]
test_input_gray=image_data_gray[0:10,:,:,:]

weight1=tf.get_variable("weight1",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,1,64])
weight2=tf.get_variable("weight2",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,64,64])
weight3=tf.get_variable("weight3",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,64,128])
weight4=tf.get_variable("weight4",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,128,128])
weight5=tf.get_variable("weight5",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,128,256])
weight6=tf.get_variable("weight6",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,256,256])
weight7=tf.get_variable("weight7",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,256,512])
weight8=tf.get_variable("weight8",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,512,256])
weight9=tf.get_variable("weight9",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,256,128])
weight10=tf.get_variable("weight10",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,128,64])
weight11=tf.get_variable("weight11",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,64,32])
weight12=tf.get_variable("weight12",initializer=tf.contrib.layers.xavier_initializer(),shape=[3,3,32,3])

input_color_data_tensor=tf.placeholder(tf.float32,shape=[None,256,256,3])
input_gray_data_tensor=tf.placeholder(tf.float32,shape=[None,256,256,1])
convlayer1=tf.nn.conv2d(input_gray_data_tensor,weight1,strides=[1,1,1,1],padding="SAME",name="conv_layer1")
activate_layer1=tf.nn.relu(convlayer1)
convlayer2=tf.nn.conv2d(activate_layer1,weight2,strides=[1,2,2,1],padding="SAME",name="conv_layer2")
activate_layer2=tf.nn.relu(convlayer2)
convlayer3=tf.nn.conv2d(activate_layer2,weight3,strides=[1,1,1,1],padding="SAME",name="conv_layer3")
activate_layer3=tf.nn.relu(convlayer3)
convlayer4=tf.nn.conv2d(activate_layer3,weight4,strides=[1,2,2,1],padding="SAME",name="conv_layer4")
activate_layer4=tf.nn.relu(convlayer4)
convlayer5=tf.nn.conv2d(activate_layer4,weight5,strides=[1,1,1,1],padding="SAME",name="conv_layer5")
activate_layer5=tf.nn.relu(convlayer5)
convlayer6=tf.nn.conv2d(activate_layer5,weight6,strides=[1,2,2,1],padding="SAME",name="conv_layer6")
activate_layer6=tf.nn.relu(convlayer6)
convlayer7=tf.nn.conv2d(activate_layer6,weight7,strides=[1,1,1,1],padding="SAME",name="conv_layer7")
activate_layer7=tf.nn.relu(convlayer7)
convlayer8=tf.nn.conv2d(activate_layer7,weight8,strides=[1,1,1,1],padding="SAME",name="conv_layer8")
activate_layer8=tf.nn.relu(convlayer8)
convlayer9=tf.nn.conv2d(activate_layer8,weight9,strides=[1,1,1,1],padding="SAME")
activate_layer9=tf.nn.relu(convlayer9)
upscale_layer1=tf.image.resize_images(activate_layer9,size=(64,64))
convlayer10=tf.nn.conv2d(upscale_layer1,weight10,strides=[1,1,1,1],padding="SAME")
activate_layer10=tf.nn.relu(convlayer10)
upscale_layer2=tf.image.resize_images(activate_layer10,size=(128,128))
convlayer11=tf.nn.conv2d(upscale_layer2,weight11,strides=[1,1,1,1],padding="SAME")
activate_layer11=tf.nn.relu(convlayer11)
convlayer12=tf.nn.conv2d(activate_layer11,weight12,strides=[1,1,1,1],padding="SAME")
activate_layer12=tf.nn.tanh(convlayer12)
upsample_layer3=tf.image.resize_images(activate_layer12,(256,256))

loss_function=tf.reduce_mean(tf.squared_difference(upsample_layer3,input_color_data_tensor))
optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss_function)

saver=tf.train.Saver()

batch_size=10
epoch=20000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_count=image_data_color.shape[0]//batch_size
    for iterate in range(epoch):
        loss_list=[]
        for batch in tqdm(range(batch_count)):
            _,train_loss=sess.run([optimizer,loss_function],feed_dict={input_color_data_tensor:train_input_color[(batch*batch_size):(batch*batch_size)+batch_size,:,:,:],input_gray_data_tensor:train_input_gray[(batch*batch_size):(batch*batch_size)+batch_size,:,:,:]})
            loss_list.append(train_loss)
        output,input=sess.run([upsample_layer3,input_color_data_tensor],feed_dict={input_gray_data_tensor:test_input_gray[0:1,:,:,:],input_color_data_tensor:test_input_color[0:1,:,:,:]})
        cv2.imwrite("waste_colorize_kl/epoch_"+str(iterate)+".jpg",(output[0]+1)*127)
        print ("epoch",iterate,"loss",sum(loss_list)/batch_count)
        saver.save(sess,"weight_colorize_kl/epoch_"+str(iterate)+".ckpt")