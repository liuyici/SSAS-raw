# import network
# from dataloader import *
# from torch.utils.data import TensorDataset, DataLoader
# import torch.nn as nn
# import lr_schedule
# import utils
# import torch.nn.functional as F
# from modules import PseudoLabeledData, load_seed, load_seed_iv, split_data, z_score, normalize, load_seed_for_domain
# import numpy as np
# import adversarial
# from utils import ConditionalEntropyLoss, augment
# from models import EMA
# from cmd_1 import CMD
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
# from sklearn.preprocessing import label_binarize
# import Adver_network
# import os

# def zhanbi(labels,args):
#     target_domain =  args.target - 1
#     # 使用 numpy 库计算每个类别的占比
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     # 计算总数
#     total_samples = len(labels)
#     # 计算每个类别的占比
#     proportions = counts / total_samples
#     count_num = np.zeros(15)
#     # count_num[target_domain] = -1#标识出目标域，权重不变
#     # 打印结果
#     for label, count, proportion in zip(unique_labels, counts, proportions):
#         if label >= target_domain:
#             label += 1
#         count_num[label] = count
#         print(f"Label {label}: 计数 {count}, 占比 {proportion:.2%}")
#     # 指定归一化的范围
#     min_val = 0.5
#     max_val = 2.0
    
#     # 计算原始计数数组的最小值和最大值
#     #将数组中的零替换为 NaN
#     # your_array_without_zeros = np.where(count_num == 0, np.nan, count_num)
#     #计算除0外的最小值
#     counts_min = np.min(count_num)
#     counts_max = np.max(count_num)
    
#     # 对计数数组进行归一化
#     normalized_weights = min_val + (max_val - min_val) * (count_num - counts_min) / (counts_max - counts_min)
#     normalized_weights[target_domain] = 1#标识出目标域，权重不变
#     return normalized_weights#这里需要输出原始的索引对应的领域的权重值
 
# def test_muda(dataset_test, model,args):
#     start_test = True
#     features = None
#     new_shape = (200, 62, 9 * 5)
#     with torch.no_grad():

#         for batch_idx, data in enumerate(dataset_test):
#             Tx = data['Tx']
#             Ty = data['Ty']
#             Tx = Tx.float().cuda()
#             # tmp_Tx = Tx.reshape(*new_shape)
#             # tmp_x = augment(tmp_Tx).cuda()
#             # obtain predictions
#             feats, outputs, _ = model(Tx)

#             # concatenate predictions
#             if start_test:
#                 all_output = outputs.float().cpu()
#                 all_label = Ty.float()
#                 features = feats.float().cpu()
#                 start_test = False
#             else:
#                 all_output = torch.cat((all_output, outputs.float().cpu()), 0)
#                 all_label = torch.cat((all_label, Ty.float()), 0)
#                 features = np.concatenate((features, feats.float().cpu()), 0)

#             # obtain labels
#         _, predictions = torch.max(all_output, 1)
#         # calculate accuracy for all examples
#         accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

#         y_true = all_label.cpu().data.numpy()
#         y_pred = predictions.cpu().data.numpy()
#         labels = np.unique(y_true)

#         # Binarize ytest with shape (n_samples, n_classes)
#         ytest = label_binarize(y_true, classes=labels)
#         ypreds = label_binarize(y_pred, classes=labels)

#         # f1 = f1_score(y_true, y_pred, average='macro')
#         # auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
#         # matrix = confusion_matrix(y_true, y_pred)

#         return accuracy, features, y_pred


# def SSDA(args):#Select Source Domain Algorithm(SSDA)
#     """
#     Parameters:
#         @args: arguments
#     通过对抗学习来最大化分布差异性和最小化领域标签的交叉熵损失, 
#     从而训练出具有领域差异性的特征，
#     再有这些特征对目标域数据分类，
#     得到的分类标签的比例即是源域数据的权重矩阵，
#     输出经过权重矩阵的源域数据卖给下一阶段的训练做准备。
    
#     书写代码的第一步: 将数据处理的过程中标签Y改为领域标签,
#     书写代码的第二步：给模型添加梯度反转层，为分布最大化做准备,注意，是在将为过后的特征上添加梯度反转层
#     书写代码的第三步：输出训练好的特征,使用test_muda进行分类,
#     得到分类标签后,还要添加一个函数Label conversion algorithm(LCA)，这个函数是将分类标签转化成权重矩阵，
#     得到新的源域数据后,则是使用MFA_LR模型来进行训练。
#     """
#     # --------------------------
#     # Prepare data
#     # --------------------------
#     # Load SEED and SEED-IV data
#     if args.dataset in ["seed", "seed-iv"]:
#         print("DATA:", args.dataset, " SESSION:", args.session)
#         # Load imagined speech data
#         if args.dataset == "seed":
#             X, Y, ture_Y = load_seed_for_domain(args.file_path, session=args.session, feature="de_LDS")
#         else:
#             # [1 session]
#             if args.mixed_sessions == 'per_session':
#                 X, Y = load_seed_iv(args.file_path, session=args.session)
#             # [3 sessions]
#             elif args.mixed_sessions == 'mixed':
#                 X1, Y1 = load_seed_iv(args.file_path, session=1)
#                 X2, Y2 = load_seed_iv(args.file_path, session=2)
#                 X3, Y3 = load_seed_iv(args.file_path, session=3)

#                 X = {}
#                 Y = {}
#                 for key in X1.keys():
#                     X1[key], _, _ = z_score(X1[key])
#                     X2[key], _, _ = z_score(X2[key])
#                     X3[key], _, _ = z_score(X3[key])

#                     X[key] = np.concatenate((X1[key], X2[key], X3[key]), axis=0)
#                     Y[key] = np.concatenate((Y1[key], Y2[key], Y3[key]), axis=0)
#             else:
#                 print("Option [mixed_sessions] is not valid.")
#                 exit(-1)

#         # select target subject
#         trg_subj = args.target - 1
#         # Target data
#         Tx = np.array(X[trg_subj])
#         Ty = np.array(Y[trg_subj])

#         # subjects
#         subject_ids = X.keys()
#         num_domains = len(subject_ids)

#         # [Option 1]: Evaluation over all target domain
#         Vx = Tx
#         Vy = Ty


#         Tx, m, std = z_score(Tx)
#         Vx = normalize(Vx, mean=m, std=std)
 
#         print("Target subject:", trg_subj)
#         print("Tx:", Tx.shape, " Ty:", Ty.shape)
#         print("Vx:", Vx.shape, " Vy:", Vy.shape)
#         print("Num. domains:", num_domains)

#         print("Data were succesfully loaded")

#         # Train dataset
#         train_loader = UnalignedDataLoader()
#         train_loader.initialize(num_domains, X, Y, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
#         datasets = train_loader.load_data()


#         # Test dataset
#         test_loader = UnalignedDataLoaderTesting()
#         test_loader.initialize(Vx, Vy, 200, shuffle_testing=False, drop_last_testing=False)
#         dataset_test = test_loader.load_data()

#     else:
#         print("This dataset does not exist.")
#         exit(-1)


#     # --------------------------
#     # Create Deep Neural Network
#     # --------------------------
#     # For synthetic dataset
#     if args.dataset in ["seed", "seed-iv"]:
#         # Define Neural Network
#         # 2790 for SEED
#         # 620 for SEED-IV
#         input_size = 2790 if args.dataset == "seed" else 620   # windows_size=9
#         hidden_size = 320
#         # model = network.DFN(input_size=input_size, hidden_size=hidden_size, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class2, radius=args.radius).cuda()

#         model = network.NEWDFN(input_size=input_size, hidden_size=hidden_size, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class2, radius=args.radius).cuda()


#     else:
#         print("A neural network for this dataset has not been selected yet.")
#         exit(-1)

#     #
#     parameter_classifier = [model.get_parameters()[2]]
#     parameter_feature = model.get_parameters()[0:1]

#     optimizer_classifier = torch.optim.SGD(parameter_classifier, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)
#     optimizer_feature = torch.optim.SGD(parameter_feature, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)

#     # if gpus are availables
#     gpus = args.gpu_id.split(',')
#     if len(gpus) > 1:
#         model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

#     # ------------------------
#     # Model training
#     # ------------------------

#     # Number of centroids for semantic loss
#     if args.dataset in ["seed", "seed-iv"]:
#         Cs_memory = []
#         for d in range(num_domains):
#             Cs_memory.append(torch.zeros(args.num_class, args.bottleneck_dim).cuda())
#         Ct_memory = torch.zeros(args.num_class, args.bottleneck_dim).cuda()

#     else:
#         print("SETTING number of centroids: The dataset does not exist.")
#         exit()

#     log_total_loss = []

#     for i in range(args.max_iter1):

#         for batch_idx, data in enumerate(datasets):
#             # get the source batches
#             x_src = list()
#             y_src = list()
#             pre_y_src = list()
#             # new_shape = (args.batch_size, 62, 9 * 5)
#             index = 0
#             #列表存储每个源域的批次数据=====================================================这里改特征：切空间特征========================================================
#             for domain_idx in range(num_domains - 1):
#                 tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
#                 tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
#                 labels = torch.from_numpy(np.array([[index] * args.batch_size]).T).type(torch.FloatTensor).flatten().long().cuda()
#                 x_src.append(tmp_x)
#                 y_src.append(labels)
#                 pre_y_src.append(tmp_y)
#                 index += 1
           
#             x_trg = data['Tx'].float().cuda()
           
#             model.train(True)

#             # obtain schedule for learning rate
#             optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier, i, lr=args.lr_a)
#             optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i, lr=args.lr_a)
#             # optimizer = torch.optim.SGD(net.parameters(), lr=0.0009575, momentum=0.9)
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_feature, 'min',factor=0.35,verbose=1,min_lr=0.1,patience=5)
#             # Get features target
#             features_target, outputs_target, _ = model(x_trg)
#             # pseudo-labels
#             pseu_labels_target = torch.argmax(outputs_target, dim=1)


        
#             pred_src = []
#             mmd_b_loss = 0 
#             mmd_t_loss = 0

#             for domain_idx in range(num_domains - 1):
#                 # get features and predictions
#                 features_source, outputs_source, _ = model(x_src[domain_idx])
#                 features_s_Adver = Adver_network.ReverseLayerF.apply(features_source, args.gamma)#用这个替代features_source经过了反转层
#                 pred_src.append(outputs_source)
#                 mmd_b_loss += utils.marginal(features_s_Adver,features_target)
#                 # mmd_t_loss += utils.conditional(
#                 #     features_s_Adver,
#                 #     features_target,
#                 #     y_src[domain_idx].reshape((args.batch_size, 1)),
#                 #     torch.nn.functional.softmax(outputs_target,dim = 1),
#                 #     2.0,
#                 #     5,
#                 #     None,
#                 #     args)
#             # Stack/Concat data from each source domain
#             pred_source = torch.cat(pred_src, dim=0)
#             labels_source = torch.cat(y_src, dim=0)

#             # [COARSE-grained training loss]
#             classifier_loss = nn.CrossEntropyLoss()(pred_source, labels_source)

          
#             MMD_loss = 0.5*mmd_b_loss
           
#             total_loss = classifier_loss   + MMD_loss #一个交叉熵加上CMD、SM的领域自适应损失，再加上一个目标域的损失

#             # Reset gradients
#             optimizer_classifier.zero_grad()
#             optimizer_feature.zero_grad()

#             # Compute gradients
#             # [normal]
#             total_loss.backward()

#             # [Update weights]
#             # classifier
#             optimizer_classifier.step()
#             optimizer_feature.step()


#         # set model to test
#         model.train(False)

#         # calculate accuracy performance
#         best_acc, features, labels = test_muda(dataset_test, model,args)
#         # print("预测标签:",labels)
#         count_num = zhanbi(labels,args)
        
#         save_path = r'E:/Research/code/MFA-LR-main/count'
#         file_name = f'count_num_{trg_subj}.npy'
#         file_path = os.path.join(save_path, file_name)
#         np.save(file_path, count_num)
        
        
#         log_str = "iter: {:04f}, \t 总损失: {:.4f}".format(i, total_loss)
#         args.log_file.write(log_str)
#         args.log_file.flush()
#         print(log_str)
#         log_total_loss.append(total_loss.data)

#     return X, ture_Y, labels, model, log_total_loss, count_num


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
from utils import ConditionalEntropyLoss, augment, LabelSmooth
from models import EMA
from cmd_1 import CMD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
import Adver_network
import os

def zhanbi(labels,args):
    target_domain =  args.target - 1
    # 使用 numpy 库计算每个类别的占比
    unique_labels, counts = np.unique(labels, return_counts=True)
    # 计算总数
    total_samples = len(labels)
    # 计算每个类别的占比
    proportions = counts / total_samples
    count_num = np.zeros(15)
    # count_num[target_domain] = -1#标识出目标域，权重不变
    # 打印结果
    for label, count, proportion in zip(unique_labels, counts, proportions):
        if label >= target_domain:
            label += 1
        count_num[label] = count
        print(f"Label {label}: 计数 {count}, 占比 {proportion:.2%}")
    # 指定归一化的范围
    min_val = 0.5
    max_val = 2.0
    
    # 计算原始计数数组的最小值和最大值
    #将数组中的零替换为 NaN
    # your_array_without_zeros = np.where(count_num == 0, np.nan, count_num)
    #计算除0外的最小值
    counts_min = np.min(count_num)
    counts_max = np.max(count_num)
    
    # 对计数数组进行归一化
    normalized_weights = min_val + (max_val - min_val) * (count_num - counts_min) / (counts_max - counts_min)
    normalized_weights[target_domain] = 1#标识出目标域，权重不变
    return normalized_weights#这里需要输出原始的索引对应的领域的权重值
 
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
            feats, outputs = model(Tx)

            # concatenate predictions
            if start_test:
                all_output = outputs.float().cpu()
                all_label = Ty.float()
                features = feats.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, Ty.float()), 0)
                features = np.concatenate((features, feats.float().cpu()), 0)

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

        # f1 = f1_score(y_true, y_pred, average='macro')
        # auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        # matrix = confusion_matrix(y_true, y_pred)

        return accuracy, features, y_pred


def SSDA(args):#Select Source Domain Algorithm(SSDA)
    """
    Parameters:
        @args: arguments
    通过对抗学习来最大化分布差异性和最小化领域标签的交叉熵损失, 
    从而训练出具有领域差异性的特征，
    再有这些特征对目标域数据分类，
    得到的分类标签的比例即是源域数据的权重矩阵，
    输出经过权重矩阵的源域数据卖给下一阶段的训练做准备。
    
    书写代码的第一步: 将数据处理的过程中标签Y改为领域标签,
    书写代码的第二步：给模型添加梯度反转层，为分布最大化做准备,注意，是在将为过后的特征上添加梯度反转层
    书写代码的第三步：输出训练好的特征,使用test_muda进行分类,
    得到分类标签后,还要添加一个函数Label conversion algorithm(LCA)，这个函数是将分类标签转化成权重矩阵，
    得到新的源域数据后,则是使用MFA_LR模型来进行训练。
    """
    # --------------------------
    # Prepare data
    # --------------------------
    # Load SEED and SEED-IV data
    if args.dataset in ["seed", "seed-iv"]:
        print("DATA:", args.dataset, " SESSION:", args.session)
        # Load imagined speech data
        if args.dataset == "seed":
            X, Y, ture_Y = load_seed_for_domain(args.file_path, session=args.session, feature="de_LDS")
        else:
            # [1 session]
            if args.mixed_sessions == 'per_session':
                X, Y = load_seed_iv(args.file_path, session=args.session)
            # [3 sessions]
            elif args.mixed_sessions == 'mixed':
                X1, Y1 = load_seed_iv(args.file_path, session=1)
                X2, Y2 = load_seed_iv(args.file_path, session=2)
                X3, Y3 = load_seed_iv(args.file_path, session=3)

                X = {}
                Y = {}
                for key in X1.keys():
                    X1[key], _, _ = z_score(X1[key])
                    X2[key], _, _ = z_score(X2[key])
                    X3[key], _, _ = z_score(X3[key])

                    X[key] = np.concatenate((X1[key], X2[key], X3[key]), axis=0)
                    Y[key] = np.concatenate((Y1[key], Y2[key], Y3[key]), axis=0)
            else:
                print("Option [mixed_sessions] is not valid.")
                exit(-1)

        # select target subject
        trg_subj = args.target - 1
        # Target data
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # subjects
        subject_ids = X.keys()
        num_domains = len(subject_ids)

        # [Option 1]: Evaluation over all target domain
        Vx = Tx
        Vy = Ty


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

    else:
        print("This dataset does not exist.")
        exit(-1)

    criterion = LabelSmooth(num_class=args.num_class).to(args.device)
    # --------------------------
    # Create Deep Neural Network
    # --------------------------
    # For synthetic dataset
    if args.dataset in ["seed", "seed-iv"]:
        # Define Neural Network
        # 2790 for SEED
        # 620 for SEED-IV
        input_size = 2790 if args.dataset == "seed" else 620   # windows_size=9
        hidden_size = 320

        model = network.DFN(input_size=input_size, hidden_size=hidden_size, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class2, radius=args.radius).cuda()


    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)

    #
    parameter_classifier = [model.get_parameters()[2]]
    parameter_feature = model.get_parameters()[0:2]

    optimizer_classifier = torch.optim.SGD(parameter_classifier, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)
    optimizer_feature = torch.optim.SGD(parameter_feature, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)

    # if gpus are availables
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    # ------------------------
    # Model training
    # ------------------------

    # Number of centroids for semantic loss
    if args.dataset in ["seed", "seed-iv"]:
        Cs_memory = []
        for d in range(num_domains):
            Cs_memory.append(torch.zeros(args.num_class, args.bottleneck_dim).cuda())
        Ct_memory = torch.zeros(args.num_class, args.bottleneck_dim).cuda()

    else:
        print("SETTING number of centroids: The dataset does not exist.")
        exit()

    log_total_loss = []

    for i in range(args.max_iter1):

        for batch_idx, data in enumerate(datasets):
            # get the source batches
            x_src = list()
            y_src = list()
            pre_y_src = list()
            # new_shape = (args.batch_size, 62, 9 * 5)
            index = 0
            #列表存储每个源域的批次数据=====================================================这里改特征：切空间特征========================================================
            for domain_idx in range(num_domains - 1):
                tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
                tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
                labels = torch.from_numpy(np.array([[index] * args.batch_size]).T).type(torch.FloatTensor).flatten().long().cuda()
                x_src.append(tmp_x)
                y_src.append(labels)
                pre_y_src.append(tmp_y)
                index += 1
           
            x_trg = data['Tx'].float().cuda()
           
            model.train(True)

            # obtain schedule for learning rate
            optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier, i, lr=args.lr_a)
            optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i, lr=args.lr_a)
            # optimizer = torch.optim.SGD(net.parameters(), lr=0.0009575, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_feature, 'min',factor=0.35,verbose=1,min_lr=0.1,patience=5)
            # Get features target
            features_target, outputs_target = model(x_trg)
            # pseudo-labels
            pseu_labels_target = torch.argmax(outputs_target, dim=1)


        
            pred_src = []
            mmd_b_loss = 0 
            mmd_t_loss = 0
            wasserstein_distance = 0
            for domain_idx in range(num_domains - 1):
                # get features and predictions
                features_source, outputs_source = model(x_src[domain_idx])
                features_s_Adver = Adver_network.ReverseLayerF.apply(features_source, args.gamma)#用这个替代features_source经过了反转层
                pred_src.append(outputs_source)
                mmd_b_loss += utils.marginal(features_s_Adver,features_target)
                # wasserstein_distance += outputs_source.mean() - outputs_target.mean()
                # mmd_t_loss += utils.conditional(
                #     features_s_Adver,
                #     features_target,
                #     y_src[domain_idx].reshape((args.batch_size, 1)),
                #     torch.nn.functional.softmax(outputs_target,dim = 1),
                #     2.0,
                #     5,
                #     None,
                #     args)
            # Stack/Concat data from each source domain
            pred_source = torch.cat(pred_src, dim=0)
            labels_source = torch.cat(y_src, dim=0)

            # [COARSE-grained training loss]
            # classifier_loss = nn.CrossEntropyLoss()(pred_source, labels_source)
            classifier_loss = criterion(pred_source, labels_source.flatten())
          
            MMD_loss = 0.5*mmd_b_loss
           
            total_loss = classifier_loss   + MMD_loss #使得领域的可分性增强，分布拉大，还需要添加一个真实标签的模糊

            # Reset gradients
            optimizer_classifier.zero_grad()
            optimizer_feature.zero_grad()

            # Compute gradients
            # [normal]
            total_loss.backward()

            # [Update weights]
            # classifier
            optimizer_classifier.step()
            optimizer_feature.step()


        # set model to test
        model.train(False)

        # calculate accuracy performance
        best_acc, features, labels = test_muda(dataset_test, model,args)
        # print("预测标签:",labels)
        count_num = zhanbi(labels,args)
        
        save_path = r'E:/Research/code/MFA-LR-main/count'
        file_name = f'count_num_{trg_subj}.npy'
        file_path = os.path.join(save_path, file_name)
        np.save(file_path, count_num)
        
        
        log_str = "iter: {:04f}, \t 总损失: {:.4f}".format(i, total_loss)
        args.log_file.write(log_str)
        args.log_file.flush()
        print(log_str)
        log_total_loss.append(total_loss.data)

    return X, ture_Y, labels, model, log_total_loss, count_num