import tensorflow as tf
from include.Model import build, get_neg
from include.Test import get_hits
from include.Load import *

import warnings
warnings.filterwarnings("ignore")

import pickle
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

'''
Follow the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

from tensorboardX import SummaryWriter
writer = SummaryWriter()

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

TF_CPP_MIN_LOG_LEVEL = 2

learning_rate = 0.001

class Config():
    parser = argparse.ArgumentParser(description='RDGCN')
    parser.add_argument('--lang', type=str, default='zh_en')
    parser.add_argument('--weight', type=float, default='0.5')
    args = parser.parse_args()
    print(args.lang, args.weight)
    language = args.lang # zh_en | ja_en | fr_en
    weight = args.weight
    e1 = 'data/' + language + '/ent_ids_1'
    e2 = 'data/' + language + '/ent_ids_2'
    ill = 'data/' + language + '/ref_ent_ids'
    kg1 = 'data/' + language + '/triples_1'
    kg2 = 'data/' + language + '/triples_2'
    epochs = 150

    hidden_dim = 200#250
    in_dim = 200#300
    out_dim = 200
#region my
    # hidden_dim = 20
    # in_dim = 20
    # out_dim = 20
#endregion
    #迭代拼接策略
    # hidden_dim = 400
    # in_dim = 400
    # out_dim = 400

    act_func = tf.nn.relu
    alpha = 0.10
    beta = 0.3
    gamma = 1.0  # margin based loss
    k = 125  # number of negative samples for each positive one
    seed = 3  # 30% of seeds



if __name__ == '__main__':


    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))

    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]

    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)

    output_layer, loss = build(Config.in_dim, Config.hidden_dim, Config.out_dim, Config.act_func, Config.alpha, Config.beta, Config.gamma,
                               Config.k, Config.language[0:2], e, train, KG1 + KG2, 1, Config.weight)
    # vec, J = training(output_layer, loss, 0.001,
    #                   Config.epochs, train, e, Config.k, test)
    # output_layer, loss, learning_rate, epochs, ILL, e, k, test)

    epochs = Config.epochs
    ILL = train
    k = Config.k

    # Initialize session
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    print('initializing...')

    sess = tf.Session()

    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.80
    # sess = tf.Session(config=sess_config)
    #

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    init = tf.global_variables_initializer()
    sess.run(init)

    # Train model
    print('running...')
    J = []
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    for i in range(epochs):
        if i % 10 == 0:
            out = sess.run(output_layer)
            neg2_left = get_neg(ILL[:, 1], out, k)
            neg_right = get_neg(ILL[:, 0], out, k)

        # Construct feed dictionary
        feeddict = {"neg_left:0": neg_left,
                    "neg_right:0": neg_right,
                    "neg2_left:0": neg2_left,
                    "neg2_right:0": neg2_right}

        # Trianing step
        _, th = sess.run([train_step, loss], feed_dict=feeddict)

        # Print results
        if i % 10 == 0:
            th, outvec = sess.run([loss, output_layer], feed_dict=feeddict)
            J.append(th)
            print('%d/%d' % (i + 1, epochs))
            print('loss=', th)
            h1_left, h1_right = get_hits(outvec, test)
            print('\n')

            writer.add_histogram('train_loss', th, i)
            writer.add_histogram('h@1_left', h1_left, i)
            writer.add_histogram('h@1_right', h1_right, i)


    outvec = sess.run(output_layer)
    sess.close()
    writer.close()

    print('Optimization Finished!')

    # print('loss:', J)

    # Testing
    print('Test Result:')
    get_hits(outvec, test)



    # Store results
    f = 'results/emb_itwe_' + str(Config.weight) + '_' + Config.language + '.pkl'
    output = open(f, 'wb')
    pickle.dump(outvec, output)
    output.close()
