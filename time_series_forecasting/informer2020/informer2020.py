import os
import sys
import ailia
import numpy as np
import json

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

from informer2020_utils import Dataset_Pred
import matplotlib.pyplot as plt


# ======================
# Parameters
# ======================

DATA_PATH = 'input.csv'
SAVE_DATA_PATH = 'output.npy'

INFORMER_ETTH1_WEIGHT_PATH = 'informer_ETTh1.onnx'
INFORMER_ETTH1_MODEL_PATH  = 'informer_ETTh1.onnx.prototxt'
INFORMER_ETTM1_WEIGHT_PATH = 'informer_ETTm1.onnx'
INFORMER_ETTM1_MODEL_PATH  = 'informer_ETTm1.onnx.prototxt'

INFORMERSTACK_ETTH1_WEIGHT_PATH = 'informerstack_ETTh1.onnx'
INFORMERSTACK_ETTH1_MODEL_PATH  = 'informerstack_ETTh1.onnx.prototxt'
INFORMERSTACK_ETTM1_WEIGHT_PATH = 'informerstack_ETTm1.onnx'
INFORMERSTACK_ETTM1_MODEL_PATH  = 'informerstack_ETTm1.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/informer2020/'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting', 
    DATA_PATH, 
    SAVE_DATA_PATH
)
parser.add_argument(
    '-m', '--model', metavar='MODEL', default="informer", 
    choices=['informer', 'informerstack']
)
parser.add_argument(
    '-d', '--data', default="ETTh1", 
    choices=['ETTh1', 'ETTm1']
)
parser.add_argument(
    '--onnx', action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def _get_data():
    timeenc = 0 if args.embed!='timeF' else 1

    freq=args.detail_freq

    data_set = Dataset_Pred(
        root_path='./',
        data_path=args.data_path,
        flag='pred',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        timeenc=timeenc,
        freq=freq,
        cols=args.cols
    )

    batch_x, batch_y, batch_x_mark, batch_y_mark = data_set[0]

    batch_x = batch_x[np.newaxis, :, :]
    batch_y = batch_y[np.newaxis, :, :]
    batch_x_mark = batch_x_mark[np.newaxis, :, :]
    batch_y_mark = batch_y_mark[np.newaxis, :, :]

    return [[batch_x, batch_y, batch_x_mark, batch_y_mark]]


def _process_one_batch(net, batch_x, batch_y, batch_x_mark, batch_y_mark):
    batch_x = batch_x.astype(np.float32)
    batch_y = batch_y.astype(np.float32)
    batch_x_mark = batch_x_mark.astype(np.float32)
    batch_y_mark = batch_y_mark.astype(np.float32)

    # decoder input
    if args.padding==0:
        dec_inp = np.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).astype(np.float32)
    elif args.padding==1:
        dec_inp = np.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).astype(np.float32)
    dec_inp = np.concatenate([batch_y[:,:args.label_len,:], dec_inp], axis=1).astype(np.float32)
    
    # encoder - decoder
    # dummy concatenation for batched input
    batch_x = np.concatenate([batch_x, batch_x], axis=0)
    batch_x_mark = np.concatenate([batch_x_mark, batch_x_mark], axis=0)
    dec_inp = np.concatenate([dec_inp, dec_inp], axis=0)
    batch_y_mark = np.concatenate([batch_y_mark, batch_y_mark], axis=0)

    if not args.onnx:
        outputs = net.predict({
            'batch_x': batch_x, 
            'batch_x_mark': batch_x_mark,
            'dec_inp': dec_inp,
            'batch_y_mark': batch_y_mark
        })[0]
    else:
        outputs = net.run(
            [
                net.get_outputs()[0].name
            ], 
            {
                net.get_inputs()[0].name: batch_x, 
                net.get_inputs()[1].name: batch_x_mark,
                net.get_inputs()[2].name: dec_inp,
                net.get_inputs()[3].name: batch_y_mark
            }
        )[0]

    outputs = outputs[0, :, :]

    f_dim = -1 if args.features=='MS' else 0
    batch_y = batch_y[:,-args.pred_len:,f_dim:]

    return outputs, batch_y


def time_series_forecasting(net):
    ### prepare dataset ###
    pred_loader = _get_data()

    ### predict ###  
    preds = []
    trues = []
    
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
        pred, true = _process_one_batch(
            net, 
            batch_x, 
            batch_y, 
            batch_x_mark, 
            batch_y_mark
        )
        preds.append(pred)
        trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    
    ### save result ###
    np.save(args.savepath, preds)
    
    ### visualize ###
    # draw OT prediction
    plt.figure()
    plt.plot(trues[0,:,-1], label='GroundTruth')
    plt.plot(preds[0,:,-1], label='Prediction')
    plt.legend()
    plt.savefig("vis_{}_{}_OT.png".format(args.model, args.data))
    # draw HUFL prediction
    plt.figure()
    plt.plot(trues[0,:,0], label='GroundTruth')
    plt.plot(preds[0,:,0], label='Prediction')
    plt.legend()
    plt.savefig("vis_{}_{}_HUFL.png".format(args.model, args.data))

    if args.write_json:
        out_data = {
            '{}_{}_OT_trues'.format(args.model, args.data): trues[0,:,-1].tolist(),
            '{}_{}_OT_preds'.format(args.model, args.data): preds[0,:,-1].tolist(),
            '{}_{}_HUFL_trues'.format(args.model, args.data): trues[0,:,0].tolist(),
            '{}_{}_HUFL_preds'.format(args.model, args.data): preds[0,:,0].tolist()
        }
        with open('output.json', 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

    logger.info('Script finished successfully.')


def main():
    # update parameters
    args.embed = 'timeF'
    args.features = 'M'
    args.target = 'OT'
    args.inverse = False
    args.cols = None
    args.num_workers = 0
    args.padding = 0
    args.use_amp = False
    args.output_attention = False
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 24
    if args.data == 'ETTm1':
        args.data_path='ETTm1.csv'
        args.freq='t'
        args.detail_freq = '15min'
    elif args.data == 'ETTh1':
        args.data_path = 'ETTh1.csv'
        args.freq = 'h'
        args.detail_freq = 'h'

    # choose model
    weight_path, model_path = None, None
    if args.model == 'informer':
        if args.data == 'ETTh1':
            weight_path = INFORMER_ETTH1_WEIGHT_PATH
            model_path  = INFORMER_ETTH1_MODEL_PATH           
        elif args.data == 'ETTm1':
            weight_path = INFORMER_ETTM1_WEIGHT_PATH
            model_path  = INFORMER_ETTM1_MODEL_PATH
    elif args.model == 'informerstack':
        if args.data == 'ETTh1':
            weight_path = INFORMERSTACK_ETTH1_WEIGHT_PATH
            model_path  = INFORMERSTACK_ETTH1_MODEL_PATH
        elif args.data == 'ETTm1':
            weight_path = INFORMERSTACK_ETTM1_WEIGHT_PATH
            model_path  = INFORMERSTACK_ETTM1_MODEL_PATH
    if weight_path == None or model_path == None:
        logger.info('Invalid model or data.')
        exit()

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # net initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=-1)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)
    
    # forecasting
    time_series_forecasting(net)


if __name__ == '__main__':
    main()
