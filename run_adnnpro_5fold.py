from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import keras.losses
import os
import sys
from evaluation import scores
import numpy as np
from sklearn.model_selection import StratifiedKFold

from adnnpro_utils import load_pro_train_data, _create_dense_net

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(threshold=np.inf)
np.random.seed(777)


def main():

    pos_data_path = str(sys.argv[1])     # "promoter_data\strain_name.fa"
    neg_data_path = str(sys.argv[2])     # "promoter_data\random_strain_name.fa"
    n_5fold = int(sys.argv[3])       # 5
    save_name = str(sys.argv[4])  # "CCMP525"

    train_data, train_label = load_pro_train_data(pos_data_path, neg_data_path)

    n_seqs_train = train_data.shape[0]
    index = np.arange(n_seqs_train)
    np.random.shuffle(index)
    train_data = train_data[index]
    train_label = train_label[index]


    recalls = []
    precisions = []
    accs = []
    aucs = []
    f1_scores = []
    mccs = []
    auprs = []
    tps = []
    fns = []
    tns = []
    fps = []
    fprs = []
    tprs = []
    spes = []
    sens = []

    for i in range(n_5fold):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
        recall = []
        precision = []
        acc = []
        auc = []
        f1_score = []
        mcc = []
        aupr = []
        tp = []
        fn = []
        tn = []
        fp = []
        fpr = []
        tpr = []
        spe = []
        sen = []
        for train_index, test_index in skf.split(train_data, train_label):
            trainX, testX = train_data[train_index], train_data[test_index]
            trainY, testY = train_label[train_index], train_label[test_index]

            predictor = _create_dense_net((1, 1000, 4), depth=34, nb_dense_block=3, growth_rate=12, nb_filter=64, dropout_rate=0.25, verbose=True)

            checkpointer = ModelCheckpoint(filepath="adnppro_trained_model/{}.h5".format(save_name), verbose=1, save_best_only=True, monitor='val_acc')
            earlystopper = EarlyStopping(monitor='val_acc', patience=30, verbose=1)

            predictor.compile(loss='binary_crossentropy', optimizer=optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

            predictor.fit(trainX, trainY, batch_size=32, epochs=100, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

            predictor.load_weights("adnppro_trained_model/{}.h5".format(save_name))
            results = predictor.predict(testX)
            result = scores(testY, results[:, 0])

            recall.append(result[0])
            precision.append(result[2])
            mcc.append(result[4])
            aupr.append(result[7])
            tp.append(result[8])
            fn.append(result[9])
            tn.append(result[10])
            fp.append(result[11])
            fpr.append(result[12])
            tpr.append(result[13])
            acc.append(result[5])
            auc.append(result[6])
            f1_score.append(result[3])
            spe.append(result[14])
            sen.append(result[15])

            keras.backend.clear_session()

        recalls.append(np.mean(recall))
        precisions.append(np.mean(precision))
        mccs.append(np.mean(mcc))
        auprs.append(np.mean(aupr))
        tps.append(np.mean(tp))
        fns.append(np.mean(fn))
        tns.append(np.mean(tn))
        fps.append(np.mean(fp))
        fprs.append(np.mean(fpr))
        tprs.append(np.mean(tpr))
        accs.append(np.mean(acc))
        aucs.append(np.mean(auc))
        f1_scores.append(np.mean(f1_score))
        spes.append(np.mean(spe))
        sens.append(np.mean(sen))

    print("Recall:%.3f" % np.mean(recalls))
    print("Precision:%.3f" % np.mean(precisions))
    print("F1:%.3f" % np.mean(f1_scores))
    print("Acc:%.3f" % np.mean(accs))
    print("AUC:%.3f" % np.mean(aucs))
    print("SPE:%.3f" % np.mean(spes))
    print("SEN:%.3f" % np.mean(sens))

    print("mcc:%.3f" % np.mean(mccs))
    print("aupr:%.3f" % np.mean(auprs))
    print("tp:%.3f" % np.mean(tps))
    print("fn:%.3f" % np.mean(fns))
    print("tn:%.3f" % np.mean(tns))
    print("fp:%.3f" % np.mean(fps))
    print("fpr:%.3f" % np.mean(fprs))
    print("tpr:%.3f" % np.mean(tprs))

    out = "adnppro_trained_model/{}_5fold.txt".format(save_name)
    with open(out, 'w') as fout:

        fout.write('Recall:{}\n'.format(np.mean(recalls)))
        fout.write('Precision:{}\n'.format(np.mean(precisions)))
        fout.write('F1:{}\n'.format(np.mean(f1_scores)))
        fout.write('Acc:{}\n'.format(np.mean(accs)))
        fout.write('AUC:{}\n'.format(np.mean(aucs)))
        fout.write('spe:{}\n'.format(np.mean(spes)))
        fout.write('\n')


if __name__ == "__main__":
    main()

