'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import warnings

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.batch_test import *
from utility.helper import *
import numpy as np
from datetime import datetime
import pickle
warnings.filterwarnings('ignore')
from time import time

def load_pretrained_data(pretrain_path):
    try:
        pretrained_weights = torch.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrained_weights = None
    return pretrained_weights

if __name__ == '__main__':

    args.device = torch.device('cuda:' + str(args.gpu_id))

    if args.mode == "train":
        t0 = time()
        """
        *********************************************************
        Train.
        """
        plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

        args.node_dropout = eval(args.node_dropout)
        args.mess_dropout = eval(args.mess_dropout)

        model = NGCF(data_generator.n_users,
                     data_generator.n_items,
                     norm_adj,
                     args).to(args.device)

        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        cur_best_pre_0, stopping_step = 0, 0
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        total_start_time = time()
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss = 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1

            for idx in range(n_batch):
                users, pos_items, neg_items = data_generator.sample()
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                               pos_items,
                                                                               neg_items,
                                                                               drop_flag=args.node_dropout_flag)

                batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                                  pos_i_g_embeddings,
                                                                                  neg_i_g_embeddings)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss

            # if (epoch + 1) % 10 != 0:
            #     if args.verbose > 0 and epoch % args.verbose == 0:
            #         perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
            #             epoch, time() - t1, loss, mf_loss, emb_loss)
            #         print(perf_str)
            #     continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            # print(users_to_test)
            ret = test(model, users_to_test, drop_flag=False)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            if args.verbose > 0:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                perf_str = '[%s] - Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (dt_string, epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0],
                            ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0],
                                                                        cur_best_pre_0,
                                                                        stopping_step,
                                                                        expected_order='acc',
                                                                        flag_step=20)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            # if should_stop:
            #     break

            # *********************************************************
            # save the user & item embeddings for pretraining.
            if args.save_flag == 1:
                latest_name = f"{args.dataset}_latest.pkl"
                if args.prefix:
                   latest_name = f'{args.prefix}_{args.dataset}_latest.pkl'
                torch.save(model.state_dict(), args.weights_path + latest_name)
                print('save the latest weights in path: ', args.weights_path + latest_name)
                if ret['recall'][0] != cur_best_pre_0:
                    continue
                best_name = f"{args.dataset}_best.pkl"
                if args.prefix:
                   best_name = f'{args.prefix}_{args.dataset}_best.pkl'
                torch.save(model.state_dict(), args.weights_path + best_name)
                print('save the best weights in path: ', args.weights_path + best_name)

        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in hit[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        print(final_perf)
        print("Total time: ", time() - total_start_time)
        loss_curve_name = f"{args.dataset}_loss.pkl"
        if args.prefix:
            loss_curve_name = f'{args.prefix}_{args.dataset}_loss.pkl'
        with open("result/" + loss_curve_name, 'wb') as f:
            pickle.dump(loss_loger, f)
    elif args.mode == "test":
        print("========== TESTING MODE ==========")
        plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

        args.node_dropout = eval(args.node_dropout)
        args.mess_dropout = eval(args.mess_dropout)
        user_testing_range = eval(args.user_test_range)
        model = NGCF(data_generator.n_users,
                     data_generator.n_items,
                     norm_adj,
                     args).to(args.device)

        pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], []
        best_name = f"{args.dataset}_best.pkl"
        if args.prefix:
            best_name = f'{args.prefix}_{args.dataset}_best.pkl'
        state_dict = load_pretrained_data(args.weights_path + best_name)
        model.load_state_dict(state_dict)

        users_to_test = list(data_generator.test_set.keys())[user_testing_range[0]:user_testing_range[1]]
        # print(users_to_test)
        ret = test(model, users_to_test, drop_flag=False)

        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                     ('\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in hit[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        print(final_perf)
    else:
        from sklearn.manifold import TSNE
        print("========== VISUALIZING MODE ==========")
        plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

        model = NGCF(data_generator.n_users,
                     data_generator.n_items,
                     norm_adj,
                     args).to(args.device)

        pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], []
        best_name = f"{args.dataset}_best.pkl"
        if args.prefix:
            best_name = f'{args.prefix}_{args.dataset}_best.pkl'
        state_dict = load_pretrained_data(args.weights_path + best_name)
        model.load_state_dict(state_dict)

        users_to_test = list(data_generator.test_set.keys())[:10]
        print(users_to_test)

        pos_items_by_u = data_generator.get_pos_items_by_users(users_to_test)
        u_g_embeddings_by_u = {}
        pos_i_g_embeddings_by_u = {}
        for u, pos_items in pos_items_by_u.items():
            u_g_embeddings, pos_i_g_embeddings, _ = model.forward([u], pos_items, [])
            u_g_embeddings_by_u[u] = u_g_embeddings
            pos_i_g_embeddings_by_u[u] = pos_i_g_embeddings

