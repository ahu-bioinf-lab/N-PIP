from scrambler.models import *
from scrambler.utils import OneHotEncoder
from adnnpro_utils import load_pro_predictor, load_pro_posdata

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(777)


def main():

    pos_data_path = str(sys.argv[1])        # "promoter_data\strain_name.fa"
    load_name = str(sys.argv[2])
    save_name = str(sys.argv[3])

    x_train, y_train = load_pro_posdata(pos_data_path)

    predictor = load_pro_predictor("adnppro_trained_model/{}.h5".format(load_name))

    pseudo_count = 1.0
    encoder = OneHotEncoder(seq_length=1000, channel_map={'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3})
    sequence_template = '$' * 1000
    onehot_template = encoder(sequence_template)[None, ...]
    x_mean = (np.sum(x_train, axis=(0, 1)) + pseudo_count) / (x_train.shape[0] + 4. * pseudo_count)

    network_config = {
        'n_groups' : 5,
        'n_resblocks_per_group' : 4,
        'n_channels' : 32,
        'window_size' : 5,
        'dilation_rates' : [1, 2, 4, 2, 1],
        'drop_rate' : 0.0,
        'norm_mode' : 'instance',
        'mask_smoothing' : True,
        'mask_smoothing_window_size' : 8,
        'mask_smoothing_std' : 1.,
        'mask_drop_scales' : [1, 5],
        'mask_min_drop_rate' : 0.0,
        'mask_max_drop_rate' : 0.5,
        'label_input' : False
    }

    scrambler = Scrambler(
        scrambler_mode='inclusion',
        input_size_x=1,
        input_size_y=1000,
        n_out_channels=4,
        input_templates=[onehot_template],
        input_backgrounds=[x_mean],
        batch_size=8,
        n_samples=16,
        sample_mode='st',
        zeropad_input=False,
        mask_dropout=False,
        network_config=network_config
    )

    x_test = x_train[0: 10]     # no usage
    y_test = y_train[0: 10]     # no usage
    n_epochs = 10
    train_history = scrambler.train(
        predictor,
        x_train,
        y_train,        # no usage
        x_test,
        y_test,
        n_epochs,
        monitor_test_indices=np.arange(32).tolist(),
        monitor_batch_freq_dict={0 : 1, 100 : 5, 500 : 10},
        nll_mode='reconstruction',
        predictor_task='classification',
        entropy_mode='target',
        entropy_bits=0.4,
        entropy_weight=1.0
    )

    scrambler.save_model("scrambler_trained_model/{}.h5".format(save_name))


if __name__ == "__main__":
    main()
