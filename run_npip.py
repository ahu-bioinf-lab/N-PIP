from keras import backend as K
import os
import sys

from adnnpro_utils import load_pro_predictor, load_pro_data
from scrambler.models import *
from scrambler.utils import OneHotEncoder, get_sequence_masks
from scrambler.visualizations import plot_dna_logo, plot_dna_importance_scores

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(threshold=np.inf)
np.random.seed(777)


def main():

    strain_name = str(sys.argv[1])      # "CCMP525"
    data_path = "promoter_data/" + str(sys.argv[2])     # "strain_name.fa"
    save_figs = sys.argv[3]

    adnppro_model_path = "adnppro_trained_model/" + strain_name + ".h5"
    scrambler_model_path = "scrambler_trained_model/" + strain_name + ".h5"

    test_data = load_pro_data(data_path)

    predictor = load_pro_predictor(adnppro_model_path)
    prediction = predictor.predict(test_data)
    np.save("npip_output/npy/label.npy", prediction)

    if save_figs is False:
        K.clear_session()

    encoder = OneHotEncoder(seq_length=1000, channel_map={'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3})
    sequence_template = '$' * 1000
    pseudo_count = 1.0
    onehot_template = encoder(sequence_template)[None, ...]
    sequence_mask = get_sequence_masks([sequence_template])[0]
    x_mean = (np.sum(test_data, axis=(0, 1)) + pseudo_count) / (test_data.shape[0] + 4. * pseudo_count)

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

    scrambler.load_model(scrambler_model_path)

    pwm, sample, importance_scores = scrambler.interpret(test_data)
    np.save("npip_output/npy/importance_pwm.npy", importance_scores[:, 0, :, :])
    np.save("npip_output/npy/importance_scores.npy", importance_scores[:, 0, :, 0])

    if save_figs is True:
        for i in range(test_data.shape[0]):

            print("Test sequence " + str(i) + ":")

            y_test_hat_ref = predictor.predict(x=[test_data[i:i + 1, ...]], batch_size=1)[0, 0]
            y_test_hat = predictor.predict(x=[sample[i, ...]], batch_size=32)[:32, 0].tolist()

            print(" - Prediction (original) = " + str(round(y_test_hat_ref, 2))[:4])
            print(" - Predictions (scrambled) = " + str([float(str(round(y_test_hat[i], 2))[:4]) for i in range(len(y_test_hat))]))

            plot_dna_logo(test_data[i, 0, :, :], sequence_template=sequence_template, figsize=(20, 0.65), plot_start=0,
                          plot_end=1000, plot_sequence_template=True, save_figs=True,
                          fig_name="test_ix_" + str(i) + "_orig_sequence")
            plot_dna_logo(pwm[i, 0, :, :], sequence_template=sequence_template, figsize=(20, 0.65), plot_start=0,
                          plot_end=1000, plot_sequence_template=True, save_figs=True,
                          fig_name="test_ix_" + str(i) + "_scrambled_pwm")
            plot_dna_importance_scores(importance_scores[i, 0, :, :].T, encoder.decode(test_data[i, 0, :, :]),
                                       figsize=(20, 0.65), score_clip=None, sequence_template=sequence_template, plot_start=0,
                                       plot_end=1000, save_figs=True,
                                       fig_name="test_ix_" + str(i) + "_scores")


if __name__ == "__main__":
    main()
