from dataloader import *
import torch.nn as nn
import lr_schedule
import utils
import torch.nn.functional as F

def train(src_data, tgt_data, model, datasets, num_domains, args):
    #
    parameter_classifier = [model.get_parameters()[2]]
    parameter_feature = model.get_parameters()[0:2]
#!!!!!!!
    optimizer_classifier = torch.optim.SGD(parameter_classifier, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)
    optimizer_feature = torch.optim.SGD(parameter_feature, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)

    # if gpus are availables
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    # ------------------------
    # Model training
    # ------------------------
    log_total_loss = []
    for i in range(args.max_iter2):
        for batch_idx, data in enumerate(datasets):
            # get the source batches
            x_src = list()
            y_src = list()

            for domain_idx in range(num_domains - 1):
                tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
                tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
                x_src.append(tmp_x)
                y_src.append(tmp_y)

           
            x_trg = data['Tx'].float().cuda()
            model.train(True)

            # obtain schedule for learning rate
            optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier, i, lr=args.lr_a)
            optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i, lr=args.lr_a) 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_feature, 'min',factor=0.35,verbose=1,min_lr=0.1,patience=5)
            # Get features target
            features_target, outputs_target_1, outputs_target_2 = model(x_trg)
            # pseudo-labels
            pseu_labels_target_1 = torch.argmax(outputs_target_1, dim=1)
            pseu_labels_target_2 = torch.argmax(outputs_target_2, dim=1)


 
            pred_src = []
            mmd_b_loss = 0 
            mmd_t_loss = 0

            for domain_idx in range(num_domains  - 1):
                features_source, outputs_source = model(x_src[domain_idx])
                pred_src.append(outputs_source)
                mmd_b_loss += utils.marginal(features_source,features_target)
                mmd_t_loss += utils.conditional(
                    features_source,
                    features_target,
                    y_src[domain_idx].reshape((args.batch_size, 1)),
                    torch.nn.functional.softmax(outputs_target,dim = 1),
                    2.0,
                    5,
                    None)
            # Stack/Concat data from each source domain
            pred_source = torch.cat(pred_src, dim=0)
            labels_source = torch.cat(y_src, dim=0)

            # [COARSE-grained training loss]
            classifier_loss = nn.CrossEntropyLoss()(pred_source, labels_source)
            MMD_loss = 0.5*mmd_b_loss + 0.5*mmd_t_loss
 
            total_loss = classifier_loss   + MMD_loss #一个交叉熵加上CMD、SM的领域自适应损失，再加上一个目标域的损失

            # Reset gradients
            optimizer_classifier.zero_grad()
            optimizer_feature.zero_grad()

            total_loss.backward()

            optimizer_classifier.step()
            optimizer_feature.step()


        # set model to test
        model.train(False)

        # calculate accuracy performance
        best_acc, best_f1, best_auc, best_mat, features, labels = test_muda(dataset_test, model,args)
        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(i, best_acc, best_f1, best_auc)
        args.log_file.write(log_str)
        args.log_file.flush()
        print(log_str)
        log_total_loss.append(total_loss.data)

    return X, Y, best_acc, best_f1, best_auc, best_mat, model, log_total_loss