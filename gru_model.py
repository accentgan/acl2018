from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from scipy.io import wavfile
from generator_gru import *
from generator import *
from discriminator import *
import numpy as np
from data_loader import read_and_decode_gpu as read_and_decode, de_emph
from bnorm import VBN
from ops import *
import timeit
import os
from conditioner import * 

class Model(object):

    def __init__(self, name='BaseModel'):
        self.name = name

    def save(self, save_path, step):
        model_name = self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.save(self.sess,
                        os.path.join(save_path, model_name))

    def load(self, save_path, model_file=None):
        if not os.path.exists(save_path):
            print('[!] Checkpoints path does not exist...')
            return False
        print('[*] Reading checkpoints...')
        if model_file is None:
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False
        else:
            ckpt_name = model_file
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(save_path, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))
        return True



class GRUGAN(Model):
    """ Speech Enhancement Generative Adversarial Network """
    def __init__(self, sess, args, devices, infer=False, name='SEGAN'):
        super(GRUGAN, self).__init__(name)
        self.args = args
        self.sess = sess
        self.keep_prob = 1.
        if infer:
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        else:
            self.keep_prob = 0.5
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.d_label_smooth = args.d_label_smooth
        self.devices = devices
        self.z_dim = args.z_dim
        self.z_depth = args.z_depth
        # type of deconv
        self.deconv_type = args.deconv_type
        # specify if use biases or not
        self.bias_downconv = args.bias_downconv
        self.bias_deconv = args.bias_deconv
        self.bias_D_conv = args.bias_D_conv
        self.accent_class = args.accent_class
        # clip D values
        self.d_clip_weights = False
        # apply VBN or regular BN?
        self.disable_vbn = False
        self.save_path = args.save_path
        # num of updates to be applied to D before G
        # this is k in original GAN paper (https://arxiv.org/abs/1406.2661)
        self.disc_updates = 1
        # set preemph factor
        self.preemph = args.preemph
        if self.preemph > 0:
            print('*** Applying pre-emphasis of {} ***'.format(self.preemph))
        else:
            print('--- No pre-emphasis applied ---')
        # canvas size
        self.canvas_size = args.canvas_size
        self.deactivated_noise = False
        # dilation factors per layer (only in atrous conv G config)
        self.g_dilated_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # num fmaps for AutoEncoder SEGAN (v1)
        self.g_enc_depths = [4,8,16, 32, 64, 64, 128, 128, 256, 256, 512, 1024,256+self.accent_class]
        self.pg_enc_depths = [1024, 512, 512, 256, 256, 128, 128, 64, 64]
        # Define D fmaps
        self.d_num_fmaps = [4,8,16, 32, 64, 64, 128, 128, 256, 256, 512,512, 1024,256, 128,32]
        self.d_phone_fmaps = [ 64, 64, 128, 256, 128, 64]
        self.init_noise_std = args.init_noise_std
        self.disc_noise_std = tf.Variable(self.init_noise_std, trainable=False)
        self.disc_noise_std_summ = scalar_summary('disc_noise_std',
                                                  self.disc_noise_std)
        self.e2e_dataset = args.e2e_dataset
        # G's supervised loss weight
        self.l1_weight = args.init_l1_weight
        self.l1_lambda = tf.Variable(self.l1_weight, trainable=False)
        self.deactivated_l1 = False
        # define the functions
        self.discriminator = discriminator
        # register G non linearity
        self.g_nl = args.g_nl
        if args.g_type == 'ae':
            self.generator = MultiGenerator(self)
        elif args.g_type == 'dwave':
            self.generator = Generator(self)
        else:
            raise ValueError('Unrecognized G type {}'.format(args.g_type))
        self.phone_gen = PhoneGenerator(self)
        self.slice_num = args.slice_num
        self.recurrent_layers = args.slice_num
        self.num_runs = args.num_runs
        self.adam = args.adam
        self.build_model(args)

    def build_model(self, config):
        all_d_grads = []
        all_g_grads = []
        all_e_grads = []
        all_dl_grads = []
        all_de_grads = []
        all_g_grad_policy = []
        all_gen_grads = []
        all_disc_grads = []
        with tf.variable_scope("optimizers") as scope:
            d_opt = tf.train.RMSPropOptimizer(config.d_learning_rate)
            g_opt = tf.train.RMSPropOptimizer(config.g_learning_rate)
            e_opt = tf.train.RMSPropOptimizer(config.g_learning_rate)
            dl_opt = tf.train.AdamOptimizer(config.d_learning_rate)
            de_opt = tf.train.AdamOptimizer(config.d_learning_rate)
            gen_opt = tf.train.RMSPropOptimizer(config.g_learning_rate)
            disc_opt = tf.train.RMSPropOptimizer(config.d_learning_rate)
            if self.adam : 
                g_policy_opt = tf.train.AdamOptimizer(config.g_policy_learning_rate)
            else :
                g_policy_opt = tf.train.SGDOptimizer(config.g_policy_learning_rate)
        with tf.variable_scope("vars") as scope:
            for idx, device in enumerate(self.devices):
                with tf.device("/%s" % device):
                    with tf.name_scope("device_%s" % idx):
                        with variables_on_gpu0():
                            self.build_model_single_gpu(idx)
                            d_grads = d_opt.compute_gradients(self.losses["d_loss"][-1],
                                                                var_list=self.d_vars)
                            disc_grads = disc_opt.compute_gradients(self.losses["d_initial_loss"][-1],
                                                                var_list=self.d_vars)
                            gen_grads = gen_opt.compute_gradients(self.losses["g_initial_loss"][-1],
                                                                var_list=self.g_vars)
                            g_grads = g_opt.compute_gradients(self.losses["g_loss"][-1],
                                                              var_list=self.g_vars)
                            e_grads = e_opt.compute_gradients(self.losses["e_loss"][-1],
                                                              var_list=self.e_vars)
                            dl_grads = dl_opt.compute_gradients(self.losses["dl_disc_loss"][-1],
                                                              var_list=self.dl_vars)
                            de_grads = dl_opt.compute_gradients(self.losses["de_disc_loss"][-1],
                                                              var_list=self.de_vars)
                            g_grad_policy = g_opt.compute_gradients(self.losses["policy_loss"][-1],
                                                              var_list=self.lstm_vars)
                            all_d_grads.append(d_grads)
                            all_g_grads.append(g_grads)
                            all_e_grads.append(e_grads)
                            all_dl_grads.append(dl_grads)
                            all_de_grads.append(de_grads)
                            all_g_grad_policy.append(g_grad_policy)
                            all_disc_grads.append(disc_grads)
                            all_gen_grads.append(gen_grads)
                            tf.get_variable_scope().reuse_variables()
        avg_d_grads = average_gradients(all_d_grads)
        avg_g_grads = average_gradients(all_g_grads)
        avg_e_grads = average_gradients(all_e_grads)
        avg_dl_grads = average_gradients(all_de_grads)
        avg_de_grads = average_gradients(all_dl_grads)
        avg_g_policy_grads = average_gradients(all_g_grad_policy)
        avg_gen_grads = average_gradients(all_gen_grads)
        avg_disc_grads = average_gradients(all_disc_grads)
        with tf.variable_scope("grad_apply",reuse=None) as scope:
            self.g_opt = g_opt.apply_gradients(avg_g_grads)
            self.d_opt = d_opt.apply_gradients(avg_d_grads)
            self.e_opt = e_opt.apply_gradients(avg_e_grads)
            self.de_opt = e_opt.apply_gradients(avg_de_grads)
            self.dl_opt = e_opt.apply_gradients(avg_dl_grads)
            self.g_policy_opt = g_policy_opt.apply_gradients(avg_g_policy_grads)
            self.gen_opt = gen_opt.apply_gradients(avg_gen_grads)
            self.disc_opt = disc_opt.apply_gradients(avg_disc_grads)

    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            # create the nodes to load for input pipeline
            filename_queue = tf.train.string_input_producer([self.e2e_dataset])
            self.get_wav, self.get_noisy, \
            self.get_val, self.get_dash, \
            self.get_embedding = read_and_decode(filename_queue,
                                   self.canvas_size,
                                   self.preemph, slice_num=self.recurrent_layers)
        # load the data to input pipeline
        wavbatch, \
        noisybatch, \
        class_val, class_dash, \
        embedding_dash = tf.train.shuffle_batch([self.get_wav,
                                             self.get_noisy, self.get_val, 
                                             self.get_dash,self.get_embedding],
                                             batch_size=self.batch_size,
                                             num_threads=2,
                                             capacity=1000 + 3 * self.batch_size,
                                             min_after_dequeue=1000,
                                             name='wav_and_noisy')
        #session = tf.InteractiveSession()
        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_wavs = []
            self.gtruth_noisy = []
            self.gtruth_val = []
            self.gtruth_dash = []
            self.embedding = []

        self.gtruth_wavs.append(wavbatch)
        self.gtruth_noisy.append(noisybatch)
        self.gtruth_val.append(class_val) 
        self.gtruth_dash.append(class_dash)
        self.embedding.append(embedding_dash)
        # add channels dimension to manipulate in D and G
        wavbatch = tf.expand_dims(wavbatch, -1)
        noisybatch = tf.expand_dims(noisybatch, -1)
        # by default leaky relu is used
        do_prelu = False
        if self.g_nl == 'prelu':
            do_prelu = True
        if gpu_idx == 0:
            #self.sample_wavs = tf.placeholder(tf.float32, [self.batch_size,
            #                                               self.canvas_size],
            #                                  name='sample_wavs')
            hidden = self.generator.zero(self.batch_size)
            ref_Gs = self.generator(noisybatch[:,0], hidden, is_ref=True,
                                    spk=None,
                                    do_prelu=do_prelu)

            print(ref_Gs[1])
            d = ClassDiscriminator(tf.squeeze(ref_Gs[1][:,:,:256]),reuse=False)
            de_r1_logits = EmbeddingDiscriminator(tf.squeeze(ref_Gs[2][:,:,:256]),reuse=False)
            self.reference_G = ref_Gs[0]
            self.ref_z = ref_Gs[1]
            if do_prelu:
                self.ref_alpha = ref_Gs[2:]
                self.alpha_summ = []
                for m, ref_alpha in enumerate(self.ref_alpha):
                    # add a summary per alpha
                    self.alpha_summ.append(histogram_summary('alpha_{}'.format(m),
                                                             ref_alpha))
            # make a dummy copy of discriminator to have variables and then
            # be able to set up the variable reuse for all other devices
            # merge along channels and this would be a real batch

            dim = noisybatch.get_shape().as_list()[-2]
            n = noisybatch.get_shape().as_list()[-3]
            dummy_joint = tf.reshape(wavbatch, shape=[self.batch_size,dim*n])
            print(class_val)
            dummy = discriminator(self, dummy_joint, zvalue=class_val,
                                 reuse=False)
            self.losses = {}
            self.loss = {}
        G = []
        Gclass = []
        log_gaussian = []
        hidden = self.generator.zero(self.batch_size)
        g = []
        gclass = []
        with tf.name_scope("policy_training_iterations"):
            for i in range(self.recurrent_layers):
                gthis, zclass, ztot, hidden_state, real_z, encode_z  = self.generator(noisybatch[:,i],
                    hidden, is_ref=False, spk=None,do_prelu=do_prelu)
                hidden = hidden_state
                g.append(gthis)
                gclass.append(zclass)
                gruns = []
                gclasses = []
                diff = tf.expand_dims(tf.squeeze(tf.abs(tf.subtract(ztot[:,:,:256], real_z))),[-1])
                trueval = tf.expand_dims(tf.squeeze(tf.abs(encode_z)), [-1])
                log_gaussian.append(tf.squeeze(tf.matmul(diff, trueval, transpose_a=True) / (1e-2)))
                for t in range(self.num_runs):
                    grun = list(g)
                    gclassrun = list(gclass)
                    for j in range(self.recurrent_layers - i - 1):
                        gthis, zclass, ztot, hidden_state, _,_  = self.generator(gthis, 
                            hidden_state, h_i=ztot, is_ref=False, spk=None,do_prelu=do_prelu,
                            modus=1)
                        grun.append(gthis)
                        gclassrun.append(zclass)
                    gruns.append(tf.concat(grun, axis=1))
                    gclasses.append(gclassrun)
                G.append(gruns)
                Gclass.append(gclasses)
        ggen = []
        with tf.name_scope("generator-encoder_training_iterations"):
            for i in range(self.recurrent_layers):
                gthis, zclass, ztot, hidden_state, _,_  = self.generator(noisybatch[:,i], hidden_state, 
                    is_ref=False, spk=None,do_prelu=do_prelu, modus=2, h_i=ztot)
                ggen.append(gthis)
        generated_gen_wave = tf.squeeze(tf.concat(ggen, axis=1))
        self.audio_sum = {}
        self.hist_sum = {}
        generated_wave = tf.squeeze(tf.concat(g, axis=1))
        self.Gs.append(generated_wave)
        self.zs.append(ztot)
        dim = noisybatch.get_shape().as_list()[-2]
        n = noisybatch.get_shape().as_list()[-3]
        noisybatch = tf.reshape(noisybatch, shape=[self.batch_size,dim*n])
        wavbatch = tf.reshape(wavbatch, shape=[self.batch_size,dim*n])
        # add new dimension to merge with other pairs
        self.logg = trueval
        D_rl_joint = noisybatch
        D_cond_joint = noisybatch
        D_fk_joint = generated_wave
        D_fk_joint_initial = generated_gen_wave
        # build rl discriminator
        d_rl_logits = discriminator(self, D_rl_joint, zvalue=class_val,reuse=True)
        d_c_logits = discriminator(self, D_cond_joint, zvalue=class_dash, reuse=True)
        d_fk_logits = discriminator(self, D_fk_joint, zvalue=zclass, reuse=True)
        d_fk_initial_logits = discriminator(self, D_fk_joint_initial, zvalue=zclass, reuse=True)
        # build fk G discriminator
        policy = {} # policy rewards
        for i in range(self.recurrent_layers-1):
            loss_tot = []
            class_loss_tot = []
            for j in range(self.num_runs):
                D_fk_spec_joint = tf.squeeze(G[i][j])
                print(G[i][j])
                print("%d %d"%(i,j))
                d_fk_spec_logits = discriminator(self, D_fk_spec_joint, zvalue=zclass, reuse=True)
                loss = {}
                self.hist_sum["d_fk_%d_%d"%(i, j)] = histogram_summary("d_fake_%d"%(i), d_fk_logits)
                self.audio_sum["gen_%d_%d"%(i, j)] = audio_summary('G_audio', G[i][j])
                self.hist_sum["gen_%d_%d"%(i,j)] = histogram_summary('G_wav', G[i][j])
                loss["g_adv_loss"] = tf.reduce_mean(tf.squared_difference(d_fk_spec_logits, 1.), axis=1)
                loss["g_l1_loss"] = self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(D_fk_spec_joint,
                                                                  wavbatch)), axis=1)
                loss["g_loss"] = loss["g_adv_loss"] + loss["g_l1_loss"]
                loss_tot.append(tf.expand_dims(loss["g_loss"],[-1]))
            loss_tot = tf.concat(loss_tot, axis=1)
            policy["layer_%d"%(i)] =tf.squeeze(tf.reduce_mean(loss_tot, axis=1))
        print("Built policy gradients")
        # class_variable_discriminator
        dl_r1_logits = ClassDiscriminator(tf.cast(class_dash, tf.float32), reuse=False)
        dl_r2_logits = ClassDiscriminator(tf.cast(class_val, tf.float32))
        dl_fk_logit_list = []
        for i in range(self.recurrent_layers):
            dl_fk_logit_list.append(ClassDiscriminator(zclass))
            self.hist_sum["dl_fk_%d"%(i)] = histogram_summary("d_latent_class", 
                dl_fk_logit_list[-1])
        # embedding discriminator 
        zval = tf.squeeze(ztot[:,:,:256])
        print(zval)
        print(embedding_dash)
        de_r1_logits = EmbeddingDiscriminator(zval,reuse=False)
        de_fk_logits = EmbeddingDiscriminator(embedding_dash)
        # make disc variables summaries
        self.hist_sum["d_rl"] = histogram_summary("d_real", d_rl_logits)
        
        self.hist_sum["de_fk"] = histogram_summary("d_latent_embedding", de_fk_logits)
        #self.d_nfk_sum = histogram_summary("d_noisyfake", d_nfk_logits)
        self.audio_sum["r1"] = audio_summary('real_audio', wavbatch)
        self.hist_sum["real"] = histogram_summary('real_wav', wavbatch)
        self.audio_sum["noisy"] = audio_summary('noisy_audio', noisybatch)
        self.hist_sum["noisy_wav"] = histogram_summary('noisy_wav', noisybatch)
           
        # handling loss functions 
        self.loss["d_rl_loss"] = tf.reduce_mean(tf.squared_difference(d_rl_logits, 1.))
        self.loss["d_fk_loss"] = tf.reduce_mean(tf.squared_difference(d_fk_logits, 0.))
        self.loss["d_c_loss"] = tf.reduce_mean(tf.squared_difference(d_c_logits, 0.))
        self.loss["d_fk_initial_loss"] = tf.reduce_mean(tf.squared_difference(d_fk_initial_logits, 0.))
        self.loss["g_adv_initial_loss"] = tf.reduce_mean(tf.squared_difference(d_fk_initial_logits, 1.))
        #d_nfk_loss = tf.reduce_mean(tf.squared_difference(d_nfk_logits, 0.))
        self.loss["g_adv_loss"] = tf.reduce_mean(tf.squared_difference(d_fk_logits, 1.))

        self.loss["d_loss"] = 2*self.loss["d_rl_loss"] + self.loss["d_fk_loss"] + self.loss["d_c_loss"]
        self.loss["d_initial_loss"] = 2*self.loss["d_rl_loss"] + self.loss["d_fk_initial_loss"] + self.loss["d_c_loss"]

        # Add the L1 loss to G
        self.loss["g_l1_loss"] = self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(generated_wave,
                                                                    wavbatch)))
        self.loss["g_l1_initial_loss"] = self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(generated_gen_wave, 
                                                                    wavbatch)))
        self.loss["g_loss"] = self.loss["g_adv_loss"] + self.loss["g_l1_loss"]
        self.loss["g_initial_loss"] = self.loss["g_l1_initial_loss"] + self.loss["g_adv_initial_loss"]
        # class_losses
        self.loss["dl_fk_loss"] = 0.
        for i in range(self.recurrent_layers):
            self.loss["dl_fk_loss"] += tf.reduce_mean(tf.squared_difference(dl_fk_logit_list[i], 0.))
        self.loss["dl_fk_loss"] /= tf.cast(self.recurrent_layers, tf.float32)
        self.loss["dl_r1_loss"] = tf.reduce_mean(tf.squared_difference(dl_r1_logits, 1.))
        self.loss["dl_r2_loss"] = tf.reduce_mean(tf.squared_difference(dl_r2_logits, 1.))
        self.loss["dl_disc_loss"] = 2*self.loss["dl_fk_loss"] + self.loss["dl_r1_loss"] + self.loss["dl_r2_loss"]
        self.loss["dl_gen_loss"] = tf.constant(0.)
        for i in range(self.recurrent_layers):
            self.loss["dl_gen_loss"] += tf.reduce_mean(tf.squared_difference(dl_fk_logit_list[i],1.))
        self.loss["dl_gen_loss"] /= tf.cast(self.recurrent_layers, tf.float32)
        # embedding-losses
        self.loss["de_r1_loss"] = tf.reduce_mean(tf.squared_difference(de_r1_logits, 1.))
        self.loss["de_fk_loss"] = tf.reduce_mean(tf.squared_difference(de_fk_logits, 0.))
        self.loss["de_disc_loss"] = self.loss["de_r1_loss"] + self.loss["de_fk_loss"]
        self.loss["de_gen_loss"] = tf.reduce_mean(tf.squared_difference(de_fk_logits, 1.))
        # encoder losses
        self.loss["e_loss"] = self.loss["de_gen_loss"] + self.loss["dl_gen_loss"]
        self.loss["dl_loss"] = self.loss["de_disc_loss"]
        self.loss["dc_loss"] = self.loss["dl_disc_loss"]
        # loseses tracking
        self.loss["policy_loss"] = tf.constant(0.)
        for i in range(self.recurrent_layers - 1):
            self.loss["policy_loss"] = self.loss["policy_loss"] + (policy["layer_%d"%(i)] * 
                log_gaussian[i])
        self.loss["policy_loss"] = tf.reduce_mean(self.loss["policy_loss"])
        self.loss_sum = {}
        for i in self.loss.keys():
            if gpu_idx == 0:
                self.losses[i] = []
            self.losses[i].append(self.loss[i])
            self.loss_sum[i] = scalar_summary(i, self.loss[i])

        if gpu_idx == 0:
            self.get_vars()




    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.d_vars_dict = {}
        self.g_vars_dict = {}
        self.e_vars_dict = {}
        self.dl_vars_dict = {}
        self.de_vars_dict = {}
        self.lstm_vars_dict = {}
        for i in range(len(self.devices)):
            for var in t_vars:
                if var.name.startswith('vars/d_'):
                    self.d_vars_dict[var.name] = var
                if var.name.startswith('vars/g_'):
                    self.g_vars_dict[var.name] = var
                if var.name.startswith('vars/g_e'):
                    self.e_vars_dict[var.name] = var
                if var.name.startswith('vars/c_d_latent_embedding'):
                    self.de_vars_dict[var.name] = var
                if var.name.startswith('vars/c_d_latent_class'):
                    self.dl_vars_dict[var.name] = var
                if var.name.startswith('vars/g_gru'):
                    self.lstm_vars_dict[var.name] = var
        self.d_vars = self.d_vars_dict.values()
        self.g_vars = self.g_vars_dict.values()
        self.e_vars = self.e_vars_dict.values()
        self.de_vars = self.de_vars_dict.values()
        self.dl_vars = self.dl_vars_dict.values()
        self.lstm_vars = self.lstm_vars_dict.values()
        self.g_vars = list(set(self.g_vars) - set(self.lstm_vars))
        self.lstm_vars += list(set(self.g_vars) - set(self.e_vars))
        for x in self.d_vars:
            assert x not in self.g_vars
        for x in self.g_vars:
            assert x not in self.d_vars
        self.all_vars = t_vars
        if self.d_clip_weights:
            print('Clipping D weights')
            self.d_clip = [v.assign(tf.clip_by_value(v, -0.05, 0.05)) for v in self.d_vars]
        else:
            print('Not clipping D weights')

    def vbn(self, tensor, name):
        if self.disable_vbn:
            class Dummy(object):
                # Do nothing here, no bnorm
                def __init__(self, tensor, ignored):
                    self.reference_output=tensor
                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def train(self, config, devices):
        """ Train the SEGAN """

        print('Initializing optimizers...')
        # init optimizers
        d_opt = self.d_opt
        g_opt = self.g_opt
        e_opt = self.e_opt
        de_opt = self.de_opt
        dl_opt = self.dl_opt
        policy_opt = self.g_policy_opt
        disc_opt = self.disc_opt
        gen_opt = self.gen_opt
        num_devices = len(devices)

        try:
            init = tf.global_variables_initializer()
        except AttributeError:
            # fall back to old implementation
            init = tf.initialize_all_variables()

        print('Initializing variables...')
        self.sess.run(init)
        g_summs = self.loss_sum.values()
        # if we have prelus, add them to summary
        if hasattr(self, 'alpha_summ'):
            g_summs += self.alpha_summ
        self.g_sum = tf.summary.merge(g_summs)
        # self.d_sum = tf.summary.merge()

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.summary.FileWriter(os.path.join(config.save_path,
                                                         'train'),
                                            self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('Sampling some wavs to store sample references...')
        # Hang onto a copy of wavs so we can feed the same one every time
        # we store samples to disk for hearing
        # pick a single batch
        sample_noisy, sample_wav, \
        sample_z = self.sess.run([self.gtruth_noisy[0],
                                  self.gtruth_wavs[0],
                                  self.zs[0]])
        print('sample noisy shape: ', sample_noisy.shape)
        print('sample wav shape: ', sample_wav.shape)
        print('sample z shape: ', sample_z.shape)

        save_path = config.save_path
        counter = 0
        # count number of samples
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.e2e_dataset):
            if num_examples % 100 == 0 : print(num_examples,end='\r')
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset,
                                                          num_examples))
        # last samples (those not filling a complete batch) are discarded
        num_batches = num_examples / self.batch_size

        print('Batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = config.start_epoch
        batch_timings = []
        d_fk_losses = []
        #d_nfk_losses = []
        d_rl_losses = []
        d_c_losses = []
        g_adv_losses = []
        g_l1_losses = []
        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                if counter % config.save_freq == 0:
                    if curr_epoch  < 4 and curr_epoch > 0:
                        for d_iter in range(self.disc_updates):
                            _d_opt, _d_sum, \
                            d_fk_loss, d_c_loss, \
                            d_rl_loss,diff,policy_loss = self.sess.run([disc_opt, self.g_sum,
                               self.losses["d_fk_loss"][0], 
                               self.losses["d_c_loss"][0],
                               self.losses["d_rl_loss"][0],
                               self.logg, self.losses["policy_loss"][0]])
                        _g_opt, _g_sum, \
                        g_adv_loss, \
                        g_l1_loss = self.sess.run([g_opt, self.g_sum,
                           self.losses["g_adv_loss"][0],
                           self.losses["g_l1_loss"][0]])
                    else :
                        for d_iter in range(self.disc_updates):
                            _d_opt, _d_sum, \
                            d_fk_loss, d_c_loss, \
                            d_rl_loss,diff = self.sess.run([d_opt, self.g_sum,
                                                       self.losses["d_fk_loss"][0], 
                                                       self.losses["d_c_loss"][0],
                                                       self.losses["d_rl_loss"][0],
                                                       self.logg])
                            _ = self.sess.run([de_opt, dl_opt])
                            if self.d_clip_weights:
                                self.sess.run(self.d_clip)
                        _policy_opt, policy_loss = self.sess.run([
                                policy_opt, self.losses["policy_loss"][0]])
                        _g_opt, _g_sum, \
                        g_adv_loss, \
                        g_l1_loss = self.sess.run([g_opt, self.g_sum,
                                                   self.losses["g_adv_loss"][0],
                                                   self.losses["g_l1_loss"][0]])
                else :
                    if curr_epoch < 4 and curr_epoch > 0:
                        for d_iter in range(self.disc_updates):
                            _d_opt, _d_sum, \
                            d_fk_loss, d_c_loss, \
                            d_rl_loss,diff = self.sess.run([disc_opt, self.g_sum,
                               self.losses["d_fk_loss"][0], 
                               self.losses["d_c_loss"][0],
                               self.losses["d_rl_loss"][0],
                               self.logg])
                        _g_opt, _g_sum, \
                        g_adv_loss, \
                        g_l1_loss = self.sess.run([g_opt, self.g_sum,
                           self.losses["g_adv_loss"][0],
                           self.losses["g_l1_loss"][0]])
                    else:
                        for d_iter in range(self.disc_updates):
                            _d_opt, \
                            d_fk_loss, d_c_loss, \
                            d_rl_loss,diff = self.sess.run([d_opt,
                                                       self.losses["d_fk_loss"][0], 
                                                       self.losses["d_c_loss"][0],
                                                       #self.d_nfk_losses[0],
                                                       self.losses["d_rl_loss"][0],
                                                       self.logg])
                            _policy_opt, policy_loss = self.sess.run([
                                policy_opt, self.losses["policy_loss"][0]])
                                                    #d_nfk_loss, \
                            if self.d_clip_weights:
                                self.sess.run(self.d_clip)

                        _g_opt, \
                        g_adv_loss, \
                        g_l1_loss = self.sess.run([policy_opt, self.losses["g_adv_loss"][0],
                                                   self.losses["g_l1_loss"][0]])
                end = timeit.default_timer()
                batch_timings.append(end - start)
                d_fk_losses.append(d_fk_loss)
                #d_nfk_losses.append(d_nfk_loss)
                d_rl_losses.append(d_rl_loss)
                d_c_losses.append(d_c_loss)
                g_adv_losses.append(g_adv_loss)
                g_l1_losses.append(g_l1_loss)
                print('{}/{} (epoch {}), d_rl_loss = {:.5f}, '
                      'd_fk_loss = {:.5f}, d_c_loss = {:.5f}, '
                      'g_adv_loss = {:.5f}, g_l1_loss = {:.5f}, policy_loss = {:.5f},'
                      ' time/batch = {:.5f}, '
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    curr_epoch,
                                                    d_rl_loss,
                                                    d_fk_loss,
                                                    d_c_loss,
                                                    g_adv_loss,
                                                    g_l1_loss,
                                                    policy_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += num_devices
                counter += num_devices
                if (counter / num_devices) % config.save_freq == 0:
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    fdict = {self.gtruth_noisy[0]:sample_noisy,
                             self.zs[0]:sample_z}
                    canvas_w = self.sess.run(self.Gs[0],
                                             feed_dict=fdict)
                    shape = [sample_wav.shape[0], sample_wav.shape[1] * sample_wav.shape[2]]
                    swaves_save = sample_wav.reshape(shape)
                    sample_dif_save = (sample_wav - sample_noisy).reshape(shape)
                    sample_noisy_save = sample_noisy.reshape(shape)
                    canvas_w_save = canvas_w.reshape(shape)
                    for m in range(min(20, canvas_w.shape[0])):
                        print('w{} max: {} min: {}'.format(m,
                                                           np.max(canvas_w_save[m]),
                                                           np.min(canvas_w_save[m])))
                        wavfile.write(os.path.join(save_path,
                                                   'sample_{}-'
                                                   '{}.wav'.format(counter, m)),
                                      config.sample_rate,
                                      de_emph(canvas_w_save[m],
                                              self.preemph))
                        m_gtruth_path = os.path.join(save_path, 'gtruth_{}.'
                                                                'wav'.format(m))
                        if not os.path.exists(m_gtruth_path):
                            wavfile.write(os.path.join(save_path,
                                                       'gtruth_{}.'
                                                       'wav'.format(m)),
                                          config.sample_rate,
                                          de_emph(swaves_save[m],
                                                  self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'noisy_{}.'
                                                       'wav'.format(m)),
                                          config.sample_rate,
                                          de_emph(sample_noisy_save[m],
                                                  self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'dif_{}.wav'.format(m)),
                                          config.sample_rate,
                                          de_emph(sample_dif_save[m],
                                                  self.preemph))
                        np.savetxt(os.path.join(save_path, 'd_rl_losses.txt'),
                                   d_rl_losses)
                        np.savetxt(os.path.join(save_path, 'd_fk_losses.txt'),
                                   d_fk_losses)
                        np.savetxt(os.path.join(save_path, 'g_adv_losses.txt'),
                                   g_adv_losses)
                        np.savetxt(os.path.join(save_path, 'policy_losses.txt'),
                                   g_l1_losses)

                if batch_idx >= num_batches:
                    curr_epoch += 1
                    # re-set batch idx
                    batch_idx = 0
                    # check if we have to deactivate L1
                    if curr_epoch >= config.l1_remove_epoch and self.deactivated_l1 == False:
                        print('** Deactivating L1 factor! **')
                        self.sess.run(tf.assign(self.l1_lambda, 0.))
                        self.deactivated_l1 = True
                    # check if we have to start decaying noise (if any)
                    if curr_epoch >= config.denoise_epoch and self.deactivated_noise == False:
                        # apply noise std decay rate
                        decay = config.noise_decay
                        if not hasattr(self, 'curr_noise_std'):
                            self.curr_noise_std = self.init_noise_std
                        new_noise_std = decay * self.curr_noise_std
                        if new_noise_std < config.denoise_lbound:
                            print('New noise std {} < lbound {}, setting 0.'.format(new_noise_std, config.denoise_lbound))
                            print('** De-activating noise layer **')
                            # it it's lower than a lower bound, cancel out completely
                            new_noise_std = 0.
                            self.deactivated_noise = True
                        else:
                            print('Applying decay {} to noise std {}: {}'.format(decay, self.curr_noise_std, new_noise_std))
                        self.sess.run(tf.assign(self.disc_noise_std, new_noise_std))
                        self.curr_noise_std = new_noise_std
                if curr_epoch >= config.epoch:
                    # done training
                    print('Done training; epoch limit {} '
                          'reached.'.format(self.epoch))
                    print('Saving last model at iteration {}'.format(counter))
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    break
        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.join(threads)
    def infer(self, config, devices):
        """ Train the SEGAN """

        print('Initializing optimizers...')
        # init optimizers
        d_opt = self.d_opt
        g_opt = self.g_opt
        e_opt = self.e_opt
        de_opt = self.de_opt
        dl_opt = self.dl_opt
        policy_opt = self.g_policy_opt
        disc_opt = self.disc_opt
        gen_opt = self.gen_opt
        num_devices = len(devices)

        try:
            init = tf.global_variables_initializer()
        except AttributeError:
            # fall back to old implementation
            init = tf.initialize_all_variables()

        print('Initializing variables...')
        self.sess.run(init)
        g_summs = self.loss_sum.values()
        # if we have prelus, add them to summary
        if hasattr(self, 'alpha_summ'):
            g_summs += self.alpha_summ
        self.g_sum = tf.summary.merge(g_summs)
        # self.d_sum = tf.summary.merge()

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.summary.FileWriter(os.path.join(config.save_path,
                                                         'train'),
                                            self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('Sampling some wavs to store sample references...')
        # Hang onto a copy of wavs so we can feed the same one every time
        # we store samples to disk for hearing
        # pick a single batch
        sample_noisy, sample_wav, \
        sample_z = self.sess.run([self.gtruth_noisy[0],
                                  self.gtruth_wavs[0],
                                  self.zs[0]])
        print('sample noisy shape: ', sample_noisy.shape)
        print('sample wav shape: ', sample_wav.shape)
        print('sample z shape: ', sample_z.shape)

        save_path = config.save_path
        counter = 0
        # count number of samples
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.e2e_dataset):
            if num_examples % 100 == 0 : print(num_examples,end='\r')
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset,
                                                          num_examples))
        # last samples (those not filling a complete batch) are discarded
        num_batches = num_examples / self.batch_size

        print('Batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = config.start_epoch
        batch_timings = []
        d_fk_losses = []
        #d_nfk_losses = []
        d_rl_losses = []
        d_c_losses = []
        g_adv_losses = []
        g_l1_losses = []
        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                end = timeit.default_timer()
                batch_timings.append(end - start)
                d_fk_losses.append(d_fk_loss)
                #d_nfk_losses.append(d_nfk_loss)
                d_rl_losses.append(d_rl_loss)
                d_c_losses.append(d_c_loss)
                g_adv_losses.append(g_adv_loss)
                g_l1_losses.append(g_l1_loss)
                print('{}/{} (epoch {}), d_rl_loss = {:.5f}, '
                      'd_fk_loss = {:.5f}, d_c_loss = {:.5f}, '
                      'g_adv_loss = {:.5f}, g_l1_loss = {:.5f}, policy_loss = {:.5f},'
                      ' time/batch = {:.5f}, '
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    curr_epoch,
                                                    d_rl_loss,
                                                    d_fk_loss,
                                                    d_c_loss,
                                                    g_adv_loss,
                                                    g_l1_loss,
                                                    policy_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += num_devices
                counter += num_devices
                # if (counter / num_devices) % config.save_freq == 0:
                self.save(config.save_path, counter)
                self.writer.add_summary(_g_sum, counter)
                self.writer.add_summary(_d_sum, counter)
                fdict = {self.gtruth_noisy[0]:sample_noisy,
                         self.zs[0]:sample_z}
                canvas_w = self.sess.run(self.Gs[0],
                                         feed_dict=fdict)
                shape = [sample_wav.shape[0], sample_wav.shape[1] * sample_wav.shape[2]]
                swaves_save = sample_wav.reshape(shape)
                sample_dif_save = (sample_wav - sample_noisy).reshape(shape)
                sample_noisy_save = sample_noisy.reshape(shape)
                canvas_w_save = canvas_w.reshape(shape)
                for m in range(min(20, canvas_w.shape[0])):
                    print('w{} max: {} min: {}'.format(m,
                                                       np.max(canvas_w_save[m]),
                                                       np.min(canvas_w_save[m])))
                    wavfile.write(os.path.join(save_path,
                                               'sample_{}-'
                                               '{}.wav'.format(counter, m)),
                                  config.sample_rate,
                                  de_emph(canvas_w_save[m],
                                          self.preemph))
                    np.savetxt(os.path.join(save_path, 'd_rl_losses.txt'),
                               d_rl_losses)
                    np.savetxt(os.path.join(save_path, 'd_fk_losses.txt'),
                               d_fk_losses)
                    np.savetxt(os.path.join(save_path, 'g_adv_losses.txt'),
                               g_adv_losses)
                    np.savetxt(os.path.join(save_path, 'policy_losses.txt'),
                               g_l1_losses)

                if batch_idx >= 1280:
                    curr_epoch += 1
                    # re-set batch idx
                    batch_idx = 0
                if curr_epoch >= config.epoch:
                    # done training
                    print('Done inferring'
                          'reached.'.format(self.epoch))
                    # print('Saving last model at iteration {}'.format(counter))
                    # self.save(config.save_path, counter)
                    # self.writer.add_summary(_g_sum, counter)
                    # self.writer.add_summary(_d_sum, counter)
                    break
        except tf.errors.OutOfRangeError:
            print('Done inferring; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.join(threads)

    def clean(self, x):
        """ clean a utterance x
            x: numpy array containing the normalized noisy waveform
        """
        c_res = None
        for beg_i in range(0, x.shape[0], self.canvas_size):
            if x.shape[0] - beg_i  < self.canvas_size:
                length = x.shape[0] - beg_i
                pad = (self.canvas_size) - length
            else:
                length = self.canvas_size
                pad = 0
            x_ = np.zeros((self.batch_size, self.canvas_size))
            if pad > 0:
                x_[0] = np.concatenate((x[beg_i:beg_i + length], np.zeros(pad)))
            else:
                x_[0] = x[beg_i:beg_i + length]
            print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
            fdict = {self.gtruth_noisy[0]:x_}
            canvas_w = self.sess.run(self.Gs[0],
                                     feed_dict=fdict)[0]
            canvas_w = canvas_w.reshape((self.canvas_size))
            print('canvas w shape: ', canvas_w.shape)
            if pad > 0:
                print('Removing padding of {} samples'.format(pad))
                # get rid of last padded samples
                canvas_w = canvas_w[:-pad]
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
        # deemphasize
        c_res = de_emph(c_res, self.preemph)
        return c_res


class SEAE(Model):
    """ Speech Enhancement Auto Encoder """
    def __init__(self, sess, args, devices, infer=False):
        self.args = args
        self.sess = sess
        self.keep_prob = 1.
        if infer:
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        else:
            self.keep_prob = 0.5
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.devices = devices
        self.save_path = args.save_path
        # canvas size
        self.canvas_size = args.canvas_size
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.e2e_dataset = args.e2e_dataset
        # define the Generator
        self.generator = AEGenerator(self)
        self.build_model(args)

    def build_model(self, config):
        all_g_grads = []
        g_opt = tf.train.AdamOptimizer(config.g_learning_rate, config.beta_1)

        for idx, device in enumerate(self.devices):
            with tf.device("/%s" % device):
                with tf.name_scope("device_%s" % idx):
                    with variables_on_gpu0():
                        self.build_model_single_gpu(idx)
                        g_grads = g_opt.compute_gradients(self.losses["g_loss"][-1],
                                                          var_list=self.g_vars)
                        all_g_grads.append(g_grads)
                        if (len(self.devices) > 1): tf.get_variable_scope().reuse_variables()
        avg_g_grads = average_gradients(all_g_grads)
        self.g_opt = g_opt.apply_gradients(avg_g_grads)


    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            # create the nodes to load for input pipeline
            filename_queue = tf.train.string_input_producer([self.e2e_dataset])
            self.get_wav, self.get_noisy, \
            self.get_val, self.get_dash = read_and_decode(filename_queue,
                                                           2 ** 14)
        # load the data to input pipeline
        wavbatch, \
        noisybatch, \
        class_val = tf.train.shuffle_batch([self.get_wav,
             self.get_noisy, self.get_val, 
             self.get_dash],
             batch_size=self.batch_size,
             num_threads=2,
             capacity=1000 + 3 * self.batch_size,
             min_after_dequeue=1000,
             name='wav_and_noisy')
        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_wavs = []
            self.gtruth_noisy = []
            self.gtruth_val = []
            self.gtruth_dash = []

        self.gtruth_wavs.append(wavbatch)
        self.gtruth_noisy.append(noisybatch)
        self.gtruth_val.append(class_val)
        # add channels dimension to manipulate in D and G
        wavbatch = tf.expand_dims(wavbatch, -1)
        noisybatch = tf.expand_dims(noisybatch, -1)
        if gpu_idx == 0:
            #self.sample_wavs = tf.placeholder(tf.float32, [self.batch_size,
            #                                               self.canvas_size],
            #                                  name='sample_wavs')
            self.reference_G = self.generator(noisybatch[0], zvalue=class_val, 
                is_ref=True,spk=None, z_on=False)
        G = self.generator(noisybatch, zvalue=class_val, is_ref=False, spk=None, z_on=False)
        print('GAE shape: ', G.get_shape())
        self.Gs.append(G)

        self.rl_audio_summ = audio_summary('real_audio', wavbatch)
        self.real_w_summ = histogram_summary('real_wav', wavbatch)
        self.noisy_audio_summ = audio_summary('noisy_audio', noisybatch)
        self.noisy_w_summ = histogram_summary('noisy_wav', noisybatch)
        self.gen_audio_summ = audio_summary('G_audio', G)
        self.gen_summ = histogram_summary('G_wav', G)

        if gpu_idx == 0:
            self.g_losses = []

        # Add the L1 loss to G
        g_loss = tf.reduce_mean(tf.abs(tf.subtract(G, wavbatch)))

        self.g_losses.append(g_loss)

        self.g_loss_sum = scalar_summary("g_loss", g_loss)

        if gpu_idx == 0:
            self.get_vars()

    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if var.name.startswith('g_')]
        for x in t_vars:
            assert x in self.g_vars, x.name
        self.all_vars = t_vars

    def train(self, config, devices):
        """ Train the SEAE """

        print('Initializing optimizer...')
        # init optimizer
        g_opt = self.g_opt
        num_devices = len(devices)

        try:
            init = tf.global_variables_initializer()
        except AttributeError:
            # fall back to old implementation
            init = tf.initialize_all_variables()

        print('Initializing variables...')
        self.sess.run(init)
        self.saver = tf.train.Saver()
        self.g_sum = tf.summary.merge([self.g_loss_sum,
                                       self.gen_summ,
                                       self.rl_audio_summ,
                                       self.real_w_summ,
                                       self.gen_audio_summ])

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.summary.FileWriter(os.path.join(config.save_path,
                                                         'train'),
                                            self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('Sampling some wavs to store sample references...')
        # Hang onto a copy of wavs so we can feed the same one every time
        # we store samples to disk for hearing
        # pick a single batch
        sample_noisy, \
        sample_wav = self.sess.run([self.gtruth_noisy[0],
                                    self.gtruth_wavs[0]])
        print('sample noisy shape: ', sample_noisy.shape)
        print('sample wav shape: ', sample_wav.shape)
        save_path = config.save_path
        counter = 0
        # count number of samples
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.e2e_dataset):
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset,
                                                          num_examples))
        # last samples (those not filling a complete batch) are discarded
        num_batches = num_examples / self.batch_size

        print('Batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        g_losses = []
        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                if counter % config.save_freq == 0:
                    # now G iterations
                    _g_opt, _g_sum, \
                    g_loss = self.sess.run([g_opt, self.g_sum,
                                            self.g_losses[0]])
                else:
                    _g_opt, \
                    g_loss = self.sess.run([g_opt, self.g_losses[0]])

                end = timeit.default_timer()
                batch_timings.append(end - start)
                g_losses.append(g_loss)
                print('{}/{} (epoch {}), g_loss = {:.5f},'
                      ' time/batch = {:.5f}, '
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    curr_epoch,
                                                    g_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += num_devices
                counter += num_devices
                if (counter / num_devices) % config.save_freq == 0:
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    fdict = {self.gtruth_noisy[0]:sample_noisy}
                    canvas_w = self.sess.run(self.Gs[0],
                                             feed_dict=fdict)
                    swaves = sample_wav
                    sample_dif = sample_wav - sample_noisy
                    for m in range(min(20, canvas_w.shape[0])):
                        print('w{} max: {} min: {}'.format(m, np.max(canvas_w[m]), np.min(canvas_w[m])))
                        wavfile.write(os.path.join(save_path, 'sample_{}-{}.wav'.format(counter, m)), config.sample_rate, canvas_w[m])
                        if not os.path.exists(os.path.join(save_path, 'gtruth_{}.wav'.format(m))):
                            wavfile.write(os.path.join(save_path, 'gtruth_{}.wav'.format(m)), config.sample_rate, swaves[m])
                            wavfile.write(os.path.join(save_path, 'noisy_{}.wav'.format(m)), config.sample_rate, sample_noisy[m])
                            wavfile.write(os.path.join(save_path, 'dif_{}.wav'.format(m)), config.sample_rate, sample_dif[m])
                        np.savetxt(os.path.join(save_path, 'g_losses.txt'), g_losses)

                if batch_idx >= num_batches:
                    curr_epoch += 1
                    # re-set batch idx
                    batch_idx = 0
                if curr_epoch >= config.epoch:
                    # done training
                    print('Done training; epoch limit {} '
                          'reached.'.format(self.epoch))
                    print('Saving last model at iteration {}'.format(counter))
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    break
        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.join(threads)
