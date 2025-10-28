import network
from dataloader import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import lr_schedule
import utils
import torch.nn.functional as F
from modules import PseudoLabeledData, load_seed, load_seed_iv, split_data, z_score, normalize, load_seed_for_domain
import numpy as np
import adversarial
from utils import discrepancy
from models import EMA
from cmd_1 import CMD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
import lmmd
import Adver_network


def test_suda(loader, model):
    start_test = True
    with torch.no_grad():
        # get iterate data
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            # get sample and label
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            # load in gpu
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels
            # obtain predictions
            _, outputs = model(inputs)
            # concatenate predictions
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # obtain labels
    _, predictions = torch.max(all_output, 1)
    # calculate accuracy for all examples
    accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

    y_true = all_label.cpu().data.numpy()
    y_pred = predictions.cpu().data.numpy()
    labels = np.unique(y_true)

    # Binarize ytest with shape (n_samples, n_classes)
    ytest = label_binarize(y_true, classes=labels)
    ypreds = label_binarize(y_pred, classes=labels)

    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
    matrix = confusion_matrix(y_true, y_pred)

    return accuracy, f1, auc, matrix


def test_muda(dataset_test, model,args):
    start_test = True
    features = None
    new_shape = (200, 62, 9 * 5)
    with torch.no_grad():

        for batch_idx, data in enumerate(dataset_test):
            Tx = data['Tx']
            Ty = data['Ty']
            Tx = Tx.float().cuda()
            # tmp_Tx = Tx.reshape(*new_shape)
            # tmp_x = augment(tmp_Tx).cuda()
            # obtain predictions
            feats, output1, output2 = model(Tx)

            # concatenate predictions
            if start_test:
                output_1 = output1.float().cpu()
                output_2 = output2.float().cpu()
                all_label = Ty.float()
                features = feats.float().cpu()
                start_test = False
            else:
                output_1 = torch.cat((output_1, output1.float().cpu()), 0)
                output_2 = torch.cat((output_2, output2.float().cpu()), 0)
                all_label = torch.cat((all_label, Ty.float()), 0)
                features = np.concatenate((features, feats.float().cpu()), 0)

            # obtain labels
        _, predictions_1 = torch.max(output_1, 1)
        _, predictions_2 = torch.max(output_2, 1)
        # calculate accuracy for all examples
        accuracy_1 = torch.sum(torch.squeeze(predictions_1).float() == all_label).item() / float(all_label.size()[0])
        accuracy_2 = torch.sum(torch.squeeze(predictions_2).float() == all_label).item() / float(all_label.size()[0])
        if accuracy_1 >= accuracy_2:
            predictions = predictions_1
        else:
            predictions = predictions_2
        y_true = all_label.cpu().data.numpy()
        y_pred = predictions.cpu().data.numpy()
        labels = np.unique(y_true)

        # Binarize ytest with shape (n_samples, n_classes)
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)

        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        matrix = confusion_matrix(y_true, y_pred)

        return [accuracy_1,accuracy_2], f1, auc, matrix, features, y_pred


def MCDA(X, Y, count_num, args):
    """
    Parameters:
        @args: arguments
    """
    # --------------------------
    # Prepare data
    # --------------------------
    # select target subject
    trg_subj = args.target - 1
    count_domain = 0
    for ij in range(len(X)):
        X[ij] = count_num[ij] * X[ij] 
    # Target data
    Tx = np.array(X[trg_subj])
    Ty = np.array(Y[trg_subj])
    subject_ids = X.keys()
    num_domains = len(subject_ids)
    Vx = Tx
    Vy = Ty

    # Standardize target data
    Tx, m, std = z_score(Tx)
    Vx = normalize(Vx, mean=m, std=std)

    print("Target subject:", trg_subj)
    print("Tx:", Tx.shape, " Ty:", Ty.shape)
    print("Vx:", Vx.shape, " Vy:", Vy.shape)
    print("Num. domains:", num_domains)
    print("Data were succesfully loaded")
    # Train dataset
    train_loader = UnalignedDataLoader()
    train_loader.initialize(num_domains, X, Y, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()
    # Test dataset
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Vx, Vy, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()

    

    # --------------------------
    # Create Deep Neural Network
    # --------------------------
    # For synthetic dataset
    if args.dataset in ["seed", "seed-iv"]:
        input_size = 2790 if args.dataset == "seed" else 620   # windows_size=9
        hidden_size = 320

        model = network.NEWDFN(input_size=input_size, hidden_size=hidden_size, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class, radius=args.radius).cuda()


    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)

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
                #
            parameter_classifier_1 = [model.get_parameters()[2]]
            parameter_classifier_2 = [model.get_parameters()[3]]
            parameter_feature = model.get_parameters()[0:1]
                #!!!!!!!
            optimizer_classifier_1 = torch.optim.SGD(parameter_classifier_1, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)
            optimizer_classifier_2 = torch.optim.SGD(parameter_classifier_2, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)
            optimizer_feature = torch.optim.SGD(parameter_feature, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)
            optimizer_classifier_1 = lr_schedule.inv_lr_scheduler(optimizer_classifier_1, i, lr=args.lr_a)
            optimizer_classifier_2 = lr_schedule.inv_lr_scheduler(optimizer_classifier_2, i, lr=args.lr_a)

            optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i, lr=args.lr_a) 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_feature, 'min',factor=0.35,verbose=1,min_lr=0.1,patience=5)
            # Get features target
            features_target, outputs_target_1, outputs_target_2 = model(x_trg)
        


 
            pred_src_1 = []
            pred_src_11 = []
            pred_src_2 = []
            pred_src_22 = []
            mmd_b_loss = 0 
            mmd_t_loss_1 = 0
            mmd_t_loss_2 = 0
            for domain_idx in range(num_domains  - 1):
                features_source, outputs_source_1, outputs_source_2 = model(x_src[domain_idx])
                pred_src_1.append(outputs_source_1)
                pred_src_2.append(outputs_source_2)
                mmd_b_loss += utils.marginal(features_source,features_target)
                # mmd_t_loss_1 += utils.conditional(
                #     features_source,
                #     features_target,
                #     y_src[domain_idx].reshape((args.batch_size, 1)),
                #     torch.nn.functional.softmax(outputs_target_1,dim = 1),
                #     2.0,
                #     5,
                #     None)
                # mmd_t_loss_2 += utils.conditional(
                #     features_source,
                #     features_target,
                #     y_src[domain_idx].reshape((args.batch_size, 1)),
                #     torch.nn.functional.softmax(outputs_target_2,dim = 1),
                #     2.0,
                #     5,
                #     None)
            
            # Stack/Concat data from each source domain
            pred_source_1 = torch.cat(pred_src_1, dim=0)
            pred_source_2 = torch.cat(pred_src_2, dim=0)
            labels_source = torch.cat(y_src, dim=0)

            classifier_loss = nn.CrossEntropyLoss()(pred_source_1, labels_source) + nn.CrossEntropyLoss()(pred_source_2, labels_source)
            MMD_loss = mmd_b_loss 
 
            total_loss = classifier_loss   + MMD_loss #

            # Reset gradients该不该移动到最前面
            optimizer_classifier_1.zero_grad()#第一阶段，全都清空   第一阶段进行减小类别差异，显现领域差异
            optimizer_classifier_2.zero_grad()#第一阶段，全都清空
            optimizer_feature.zero_grad()#第一阶段，全都清空

            total_loss.backward()

            optimizer_classifier_1.step()#第一阶段，全都更新梯度
            optimizer_classifier_2.step()#第一阶段，全都更新梯度
            optimizer_feature.step()#第一阶段，全都更新梯度

            # Reset gradients=============================================================================================
            optimizer_classifier_1.zero_grad()#第二阶段，全都清空      第二阶段进行减小领域差异，显现类别差异
            optimizer_classifier_2.zero_grad()#第二阶段，全都清空
            optimizer_feature.zero_grad()#第二阶段，全都清空
            
            features_target_2, outputs_target_11, outputs_target_22 = model(x_trg)
            for domain_idx in range(num_domains  - 1):
                features_source, outputs_source_11, outputs_source_22 = model(x_src[domain_idx])
                pred_src_11.append(outputs_source_11)
                pred_src_22.append(outputs_source_22)
                # features_s_Adver = Adver_network.ReverseLayerF.apply(features_source, args.gamma)#用这个替代features_source经过了反转层

            # Stack/Concat data from each source domain
            pred_source_11 = torch.cat(pred_src_11, dim=0)
            pred_source_22 = torch.cat(pred_src_22, dim=0)
            labels_source_1 = torch.cat(y_src, dim=0)

            classifier_loss_1 = nn.CrossEntropyLoss()(pred_source_11, labels_source_1) + nn.CrossEntropyLoss()(pred_source_22, labels_source_1) - discrepancy(outputs_target_11, outputs_target_22)
 
            total_loss_1 = classifier_loss_1 
            
            total_loss_1.backward()

            optimizer_classifier_1.step()#第二阶段，没有更新特征提取器
            optimizer_classifier_2.step()#第二阶段，没有更新特征提取器
            # optimizer_feature.step()
            
            optimizer_classifier_1.zero_grad()#第三阶段，全都清空    第三阶段最好还是添加一个用于拉近分布的手段
            optimizer_classifier_2.zero_grad()#第三阶段，全都清空
            optimizer_feature.zero_grad()#第三阶段，全都清空
            #=========================================================================================================
            
            for index in range(1):
                features_target_end, outputs_target_1_end, outputs_target_2_end = model(x_trg)
                classifier_loss_end =  discrepancy(outputs_target_1_end, outputs_target_2_end)
     
                total_loss_end = classifier_loss_end 
                total_loss_end.backward()
                
                optimizer_feature.step()#第三阶段，仅更新特征提取器
                
                optimizer_classifier_1.zero_grad()
                optimizer_classifier_2.zero_grad()
                optimizer_feature.zero_grad()

        # set model to test
        model.train(False)

        # calculate accuracy performance
        best_acc, best_f1, best_auc, best_mat, features, labels = test_muda(dataset_test, model,args)
        log_str = "iter: {:05d}, \t accuracy: {:.4f}/{:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(i, best_acc[0], best_acc[1], best_f1, best_auc)
        args.log_file.write(log_str)
        args.log_file.flush()
        print(log_str)
        log_total_loss.append(total_loss.data)

    return X, Y, best_acc, best_f1, best_auc, best_mat, model, log_total_loss


