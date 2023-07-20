# from os.path import join

# import tensorflow as tf

# import setting
# from bak.tf1.ae_model import ae_model


# def load_model():
#     model_name = 'imageAE-8500'
#     model_dir = setting.convAE_dir
#     model_path = join(model_dir, model_name)

#     hparams = setting.synth_cnn_hparams

#     g = tf.Graph()

#     with g.as_default():
#         model = ae_model(**hparams)
#         sess = tf.Session(graph=g)

#         model['saver'].restore(sess, model_path)
#         model['sess'] = sess

#     return model
