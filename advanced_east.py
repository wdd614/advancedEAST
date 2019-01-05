import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ModelCheckpoint#callback函数
from keras.optimizers import Adam

import cfg
from network import East
from losses import quad_loss
from data_generator import gen
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

import preprocess
import label
#-----Pre-process
# preprocess.preprocess()
# label.process_label()
#----------------------
if not os.path.exists(cfg.log_dir):
    os.mkdir(cfg.log_dir)
east = East()
east_network = east.east_network()
try:#可能会出现异常情况，使用try..except消除异常情况
    east_network = multi_gpu_model(east_network, gpus=cfg.GPUs, cpu_relocation=False)
    print("=======Training using multiple GPUs..======")
except ValueError:
    # parallel_model = east_network
    print("=======Training using single GPU or CPU====")
east_network.summary()
east_network.compile(loss=quad_loss, optimizer=Adam(lr=cfg.lr,
                                                    # clipvalue=cfg.clipvalue,
                                                    decay=cfg.decay))
if cfg.load_weights and os.path.exists(cfg.saved_model_weights_file_path):
    east_network.load_weights(cfg.saved_model_weights_file_path)

east_network.fit_generator(generator=gen(),
                           steps_per_epoch=cfg.steps_per_epoch,
                           epochs=cfg.epoch_num,
                           validation_data=gen(is_val=True),
                           validation_steps=cfg.validation_steps,
                           verbose=1,
                           initial_epoch=cfg.initial_epoch,
                           callbacks=[
                               EarlyStopping(patience=cfg.patience, verbose=1),
                               ModelCheckpoint(filepath=cfg.model_weights_path,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1),
                               TensorBoard(log_dir=cfg.log_dir,#使用tensorboard进行记录
                                histogram_freq=0, write_graph=True, write_images=False),
                               ModelCheckpoint(filepath=cfg.saved_model_weights_file_path, monitor='val_loss', verbose=0,
                                                               save_best_only=True, save_weights_only=True,
                                                               mode='auto', period=1)])
east_network.save(cfg.saved_model_file_path)
east_network.save_weights(cfg.saved_model_weights_file_path)
