import sys
import argparse
import itertools as it
import traceback

import numpy as np
from scipy.special import softmax
import cv2

import torch
from torch_geometric.nn import fps
from torch_geometric.nn import radius
from torch_geometric.nn import knn_interpolate

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402

from rignet_utils import create_single_data, inside_check, sample_on_bone
from rignet_utils import meanshift_cluster, nms_meanshift, flip
from rignet_utils import increase_cost_for_outside_bone
from rignet_utils import RigInfo, TreeNode, primMST_symmetry, loadSkel_recur
from rignet_utils import get_bones, calc_geodesic_matrix, post_filter, assemble_skel_skin
from vis_utils import draw_shifted_pts, show_obj_skel

# ======================
# Parameters
# ======================

WEIGHT_JOINTNET_PATH = 'models/gcn_meanshift.onnx'
MODEL_JOINTNET_PATH = 'models/gcn_meanshift.onnx.prototxt'
WEIGHT_ROOTNET_SE_PATH = 'models/rootnet_shape_enc.onnx'
MODEL_ROOTNET_SE_PATH = 'models/rootnet_shape_enc.onnx.prototxt'
WEIGHT_ROOTNET_SA1_PATH = 'models/rootnet_sa1_conv.onnx'
MODEL_ROOTNET_SA1_PATH = 'models/rootnet_sa1_conv.onnx.prototxt'
WEIGHT_ROOTNET_SA2_PATH = 'models/rootnet_sa2_conv.onnx'
MODEL_ROOTNET_SA2_PATH = 'models/rootnet_sa2_conv.onnx.prototxt'
WEIGHT_ROOTNET_SA3_PATH = 'models/rootnet_sa3.onnx'
MODEL_ROOTNET_SA3_PATH = 'models/rootnet_sa3.onnx.prototxt'
WEIGHT_ROOTNET_FP3_PATH = 'models/rootnet_fp3_nn.onnx'
MODEL_ROOTNET_FP3_PATH = 'models/rootnet_fp3_nn.onnx.prototxt'
WEIGHT_ROOTNET_FP2_PATH = 'models/rootnet_fp2_nn.onnx'
MODEL_ROOTNET_FP2_PATH = 'models/rootnet_fp2_nn.onnx.prototxt'
WEIGHT_ROOTNET_FP1_PATH = 'models/rootnet_fp1_nn.onnx'
MODEL_ROOTNET_FP1_PATH = 'models/rootnet_fp1_nn.onnx.prototxt'
WEIGHT_ROOTNET_BL_PATH = 'models/rootnet_back_layers.onnx'
MODEL_ROOTNET_BL_PATH = 'models/rootnet_back_layers.onnx.prototxt'
WEIGHT_BONENET_SA1_PATH = 'models/bonenet_sa1_conv.onnx'
MODEL_BONENET_SA1_PATH = 'models/bonenet_sa1_conv.onnx.prototxt'
WEIGHT_BONENET_SA2_PATH = 'models/bonenet_sa2_conv.onnx'
MODEL_BONENET_SA2_PATH = 'models/bonenet_sa2_conv.onnx.prototxt'
WEIGHT_BONENET_SA3_PATH = 'models/bonenet_sa3.onnx'
MODEL_BONENET_SA3_PATH = 'models/bonenet_sa3.onnx.prototxt'
WEIGHT_BONENET_SE_PATH = 'models/bonenet_shape_enc.onnx'
MODEL_BONENET_SE_PATH = 'models/bonenet_shape_enc.onnx.prototxt'
WEIGHT_BONENET_EF_PATH = 'models/bonenet_expand_joint_feature.onnx'
MODEL_BONENET_EF_PATH = 'models/bonenet_expand_joint_feature.onnx.prototxt'
WEIGHT_BONENET_MT_PATH = 'models/bonenet_mix_transform.onnx'
MODEL_BONENET_MT_PATH = 'models/bonenet_mix_transform.onnx.prototxt'
WEIGHT_SKINNET_PATH = 'models/skinnet.onnx'
MODEL_SKINNET_PATH = 'models/skinnet.onnx.prototxt'

REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/rignet/'

INPUT_PATH = '17872_remesh.obj'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='RigNet model'
)
parser.add_argument(
    '-i', '--input', metavar='FILE',
    default=INPUT_PATH,
    help='The input .obj path.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH', default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-t', '--threshold', metavar='VAL',
    default=0.75e-5, type=float,
    help='default is %.08f.' % 0.75e-5,
)
parser.add_argument(
    '-b', '--bandwidth', metavar='VAL',
    default=0.045, type=float,
    help='default is %f.' % 0.045,
)
parser.add_argument(
    '--vox_file', metavar='FILE',
    default=None,
    help='The input .binvox path'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = parser.parse_args()


# ======================
# Secondaty Functions
# ======================

def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def geometric_fps(src, batch=None, ratio=None):
    src = torch.from_numpy(src)
    batch = torch.from_numpy(batch)
    res = fps(src, batch=batch, ratio=ratio)
    return res.numpy()


def geometric_radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    batch_x = torch.from_numpy(batch_x) if batch_x is not None else None
    batch_y = torch.from_numpy(batch_y) if batch_y is not None else None
    row, col = radius(x, y, r, batch_x, batch_y, max_num_neighbors=max_num_neighbors)
    return row.numpy(), col.numpy()


def geometric_knn_interpolate(x, pos_x, pos_y, batch_x=None, batch_y=None, k=3):
    x = torch.from_numpy(x)
    pos_x = torch.from_numpy(pos_x)
    pos_y = torch.from_numpy(pos_y)
    batch_x = torch.from_numpy(batch_x)
    batch_y = torch.from_numpy(batch_y)
    res = knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k)
    return res.numpy()


# ======================
# Main functions
# ======================


def predict_joints(
        data, vox, joint_net, threshold, bandwidth=0.04, mesh_filename=None):
    """
    Predict joints
    :param data: wrapped input data
    :param vox: voxelized mesh
    :param joint_net: network for predicting joints
    :param threshold: density threshold to filter out shifted points
    :param bandwidth: bandwidth for meanshift clustering
    :return: wrapped data with predicted joints, pair-wise bone representation added.
    """

    batch, pos, geo_edge_index, tpl_edge_index = \
        data['batch'], data['pos'], data['geo_edge_index'], data['tpl_edge_index']

    if not args.onnx:
        output = joint_net.predict({
            'batch': batch, 'pos': pos,
            'geo_edge_index': geo_edge_index, 'tpl_edge_index': tpl_edge_index
        })
    else:
        in_batch = joint_net.get_inputs()[0].name
        in_pos = joint_net.get_inputs()[1].name
        in_geo_e = joint_net.get_inputs()[2].name
        in_tpl_e = joint_net.get_inputs()[3].name
        out_displacement = joint_net.get_outputs()[0].name
        out_attn_pred0 = joint_net.get_outputs()[1].name
        out_attn_pred = joint_net.get_outputs()[2].name
        output = joint_net.run(
            [out_displacement, out_attn_pred0, out_attn_pred],
            {
                in_batch: batch, in_pos: pos,
                in_geo_e: geo_edge_index, in_tpl_e: tpl_edge_index
            })
    data_displacement, _, attn_pred = output

    y_pred = data_displacement + data['pos']
    y_pred, index_inside = inside_check(y_pred, vox)
    attn_pred = attn_pred[index_inside, :]
    y_pred = y_pred[attn_pred.squeeze() > 1e-3]
    attn_pred = attn_pred[attn_pred.squeeze() > 1e-3]

    # symmetrize points by reflecting
    y_pred_reflect = y_pred * np.array([[-1, 1, 1]])
    y_pred = np.concatenate((y_pred, y_pred_reflect), axis=0)
    attn_pred = np.tile(attn_pred, (2, 1))

    # img = draw_shifted_pts(mesh_filename, y_pred, weights=attn_pred)
    y_pred = meanshift_cluster(y_pred, bandwidth, attn_pred, max_iter=40)
    # img = draw_shifted_pts(mesh_filename, y_pred, weights=attn_pred)

    Y_dist = np.sum(((y_pred[np.newaxis, ...] - y_pred[:, np.newaxis, :]) ** 2), axis=2)
    density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
    density = np.sum(density, axis=0)
    density_sum = np.sum(density)
    y_pred = y_pred[density / density_sum > threshold]
    attn_pred = attn_pred[density / density_sum > threshold][:, 0]
    density = density[density / density_sum > threshold]

    # img = draw_shifted_pts(mesh_filename, y_pred, weights=attn_pred)
    pred_joints = nms_meanshift(y_pred, density, bandwidth)
    pred_joints, _ = flip(pred_joints)
    # img = draw_shifted_pts(mesh_filename, pred_joints)

    # prepare and add new data members
    pairs = list(it.combinations(range(pred_joints.shape[0]), 2))
    pair_attr = []
    for pr in pairs:
        dist = np.linalg.norm(pred_joints[pr[0]] - pred_joints[pr[1]])
        bone_samples = sample_on_bone(pred_joints[pr[0]], pred_joints[pr[1]])
        bone_samples_inside, _ = inside_check(bone_samples, vox)
        outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
        attr = np.array([dist, outside_proportion, 1])
        pair_attr.append(attr)

    pairs = np.array(pairs)
    pair_attr = np.array(pair_attr)
    joints_batch = np.zeros(len(pred_joints), dtype=np.int64)
    pairs_batch = np.zeros(len(pairs), dtype=np.int64)

    data['joints'] = pred_joints.astype(np.float32)
    data['pairs'] = pairs.astype(np.float32)
    data['pair_attr'] = pair_attr.astype(np.float32)
    data['joints_batch'] = joints_batch
    data['pairs_batch'] = pairs_batch
    return data


def getInitId(data, root_net):
    """
    predict root joint ID via rootnet
    :param data:
    :param model:
    :return:
    """
    batch, pos, geo_edge_index, tpl_edge_index = \
        data['batch'], data['pos'], data['geo_edge_index'], data['tpl_edge_index']
    joints, joints_batch = data['joints'], data['joints_batch']
    idx = np.random.randn(joints.shape[0]).argsort()
    joints_shuffle = joints[idx]

    # shape_encoder
    shape_encoder = root_net['shape_encoder']
    if not args.onnx:
        x_glb_shape = shape_encoder.predict({
            'batch': batch, 'pos': pos,
            'geo_edge_index': geo_edge_index, 'tpl_edge_index': tpl_edge_index
        })[0]
    else:
        in_batch = shape_encoder.get_inputs()[0].name
        in_pos = shape_encoder.get_inputs()[1].name
        in_geo_e = shape_encoder.get_inputs()[2].name
        in_tpl_e = shape_encoder.get_inputs()[3].name
        out_x_glb_shape = shape_encoder.get_outputs()[0].name
        x_glb_shape = shape_encoder.run(
            [out_x_glb_shape],
            {
                in_batch: batch, in_pos: pos,
                in_geo_e: geo_edge_index, in_tpl_e: tpl_edge_index
            })[0]
    shape_feature = np.repeat(x_glb_shape, len(joints_batch[joints_batch == 0]), axis=0)

    x = np.abs(joints_shuffle[:, 0:1])
    pos = joints_shuffle
    batch = joints_batch
    sa0_joint = (x, pos, batch)

    # sa1_joint
    ratio, r = 0.999, 0.4
    idx = geometric_fps(pos, batch, ratio=ratio)
    row, col = geometric_radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=64)
    edge_index = np.stack([col, row], axis=0)
    sa1_module = root_net['sa1_module']
    if not args.onnx:
        x = sa1_module.predict({
            'batch': batch, 'pos': pos, 'in_pos_idx': pos[idx], 'edge_index': edge_index
        })
    else:
        in_x = sa1_module.get_inputs()[0].name
        in_pos = sa1_module.get_inputs()[1].name
        in_pos_idx = sa1_module.get_inputs()[2].name
        in_edge_index = sa1_module.get_inputs()[3].name
        out = sa1_module.get_outputs()[0].name
        x = sa1_module.run(
            [out],
            {
                in_x: x, in_pos: pos, in_pos_idx: pos[idx], in_edge_index: edge_index
            })[0]
    pos, batch = pos[idx], batch[idx]
    sa1_joint = (x, pos, batch)

    # sa2_joint
    ratio, r = 0.33, 0.6
    idx = geometric_fps(pos, batch, ratio=ratio)
    row, col = geometric_radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=64)
    edge_index = np.stack([col, row], axis=0)
    sa2_module = root_net['sa2_module']
    if not args.onnx:
        x = sa2_module.predict({
            'batch': batch, 'pos': pos, 'in_pos_idx': pos[idx], 'edge_index': edge_index
        })
    else:
        in_x = sa2_module.get_inputs()[0].name
        in_pos = sa2_module.get_inputs()[1].name
        in_pos_idx = sa2_module.get_inputs()[2].name
        in_edge_index = sa2_module.get_inputs()[3].name
        out = sa2_module.get_outputs()[0].name
        x = sa2_module.run(
            [out],
            {
                in_x: x, in_pos: pos, in_pos_idx: pos[idx], in_edge_index: edge_index
            })[0]
    pos, batch = pos[idx], batch[idx]
    sa2_joint = (x, pos, batch)

    # sa3_joint
    sa3_module = root_net['sa3_module']
    if not args.onnx:
        output = sa3_module.predict({
            'batch': batch, 'pos': pos, 'batch': batch
        })
    else:
        in_x = sa3_module.get_inputs()[0].name
        in_pos = sa3_module.get_inputs()[1].name
        in_batch = sa3_module.get_inputs()[2].name
        out_x = sa3_module.get_outputs()[0].name
        out_pos = sa3_module.get_outputs()[1].name
        out_batch = sa3_module.get_outputs()[2].name
        output = sa3_module.run(
            [out_x, out_pos, out_batch],
            {
                in_x: x, in_pos: pos, in_batch: batch
            })
    sa3_joint = output

    # fp3_joint
    x, pos, batch = sa3_joint
    x_skip, pos_skip, batch_skip = sa2_joint
    x = geometric_knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=1)
    x = np.concatenate([x, x_skip], axis=1)
    fp3_module = root_net['fp3_module']
    if not args.onnx:
        x = fp3_module.predict({'x': x})[0]
    else:
        in_x = fp3_module.get_inputs()[0].name
        out_x = fp3_module.get_outputs()[0].name
        x = fp3_module.run([out_x], {in_x: x})[0]
    pos, batch = pos_skip, batch_skip

    # fp2_joint
    x_skip, pos_skip, batch_skip = sa1_joint
    x = geometric_knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=3)
    x = np.concatenate([x, x_skip], axis=1)
    fp2_module = root_net['fp2_module']
    if not args.onnx:
        x = fp2_module.predict({'x': x})[0]
    else:
        in_x = fp2_module.get_inputs()[0].name
        out_x = fp2_module.get_outputs()[0].name
        x = fp2_module.run([out_x], {in_x: x})[0]
    pos, batch = pos_skip, batch_skip

    # fp1_joint
    x_skip, pos_skip, batch_skip = sa0_joint
    x = geometric_knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=3)
    x = np.concatenate([x, x_skip], axis=1)
    fp1_module = root_net['fp1_module']
    if not args.onnx:
        joint_feature = fp1_module.predict({'x': x})[0]
    else:
        in_x = fp1_module.get_inputs()[0].name
        out_x = fp1_module.get_outputs()[0].name
        joint_feature = fp1_module.run([out_x], {in_x: x})[0]

    x_joint = np.concatenate([shape_feature, joint_feature], axis=1)

    back_layers = root_net['back_layers']
    if not args.onnx:
        x_joint = back_layers.predict({'x_joint': x_joint})[0]
    else:
        in_x = back_layers.get_inputs()[0].name
        out_x = back_layers.get_outputs()[0].name
        x_joint = back_layers.run([out_x], {in_x: x_joint})[0]

    root_prob = x_joint
    root_prob = sigmoid(root_prob)
    root_id = np.argmax(root_prob)

    return root_id


def predict_skeleton(
        data, vox, root_net, bone_net):
    """
    Predict skeleton structure based on joints
    :param data: wrapped data
    :param vox: voxelized mesh
    :param root_net: network to predict root, pairwise connectivity cost
    :param mesh_filename: meshfilename for debugging
    :return: predicted skeleton structure
    """
    root_id = getInitId(data, root_net)

    joints, pairs, pair_attr, joints_batch, pairs_batch = \
        data['joints'], data['pairs'], data['pair_attr'], data['joints_batch'], data['pairs_batch']
    sa0_joints = (None, joints, joints_batch)
    _, pos, batch = sa0_joints

    # sa1_joint
    ratio, r = 0.999, 0.4
    idx = geometric_fps(pos, batch, ratio=ratio)
    row, col = geometric_radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=64)
    edge_index = np.stack([col, row], axis=0)
    sa1_module = bone_net['sa1_module']
    if not args.onnx:
        x = sa1_module.predict({
            'pos': pos, 'in_pos_idx': pos[idx], 'edge_index': edge_index
        })
    else:
        in_pos = sa1_module.get_inputs()[0].name
        in_pos_idx = sa1_module.get_inputs()[1].name
        in_edge_index = sa1_module.get_inputs()[2].name
        out = sa1_module.get_outputs()[0].name
        x = sa1_module.run(
            [out],
            {
                in_pos: pos, in_pos_idx: pos[idx], in_edge_index: edge_index
            })[0]
    pos, batch = pos[idx], batch[idx]

    # sa2_joint
    ratio, r = 0.33, 0.6
    idx = geometric_fps(pos, batch, ratio=ratio)
    row, col = geometric_radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=64)
    edge_index = np.stack([col, row], axis=0)
    sa2_module = bone_net['sa2_module']
    if not args.onnx:
        x = sa2_module.predict({
            'batch': batch, 'pos': pos, 'in_pos_idx': pos[idx], 'edge_index': edge_index
        })
    else:
        in_x = sa2_module.get_inputs()[0].name
        in_pos = sa2_module.get_inputs()[1].name
        in_pos_idx = sa2_module.get_inputs()[2].name
        in_edge_index = sa2_module.get_inputs()[3].name
        out = sa2_module.get_outputs()[0].name
        x = sa2_module.run(
            [out],
            {
                in_x: x, in_pos: pos, in_pos_idx: pos[idx], in_edge_index: edge_index
            })[0]
    pos, batch = pos[idx], batch[idx]

    # sa3_joint
    sa3_module = bone_net['sa3_module']
    if not args.onnx:
        output = sa3_module.predict({
            'batch': batch, 'pos': pos, 'batch': batch
        })
    else:
        in_x = sa3_module.get_inputs()[0].name
        in_pos = sa3_module.get_inputs()[1].name
        in_batch = sa3_module.get_inputs()[2].name
        out_x = sa3_module.get_outputs()[0].name
        out_pos = sa3_module.get_outputs()[1].name
        out_batch = sa3_module.get_outputs()[2].name
        output = sa3_module.run(
            [out_x, out_pos, out_batch],
            {
                in_x: x, in_pos: pos, in_batch: batch
            })
    joint_feature, _, _ = output
    joint_feature = np.repeat(joint_feature, len(pairs_batch[pairs_batch == 0]), axis=0)

    batch, pos, geo_edge_index, tpl_edge_index = \
        data['batch'], data['pos'], data['geo_edge_index'], data['tpl_edge_index']

    # shape_encoder
    shape_encoder = bone_net['shape_encoder']
    if not args.onnx:
        shape_feature = shape_encoder.predict({
            'batch': batch, 'pos': pos,
            'geo_edge_index': geo_edge_index, 'tpl_edge_index': tpl_edge_index
        })[0]
    else:
        in_batch = shape_encoder.get_inputs()[0].name
        in_pos = shape_encoder.get_inputs()[1].name
        in_geo_e = shape_encoder.get_inputs()[2].name
        in_tpl_e = shape_encoder.get_inputs()[3].name
        out_shape_feature = shape_encoder.get_outputs()[0].name
        shape_feature = shape_encoder.run(
            [out_shape_feature],
            {
                in_batch: batch, in_pos: pos,
                in_geo_e: geo_edge_index, in_tpl_e: tpl_edge_index
            })[0]
    shape_feature = np.repeat(shape_feature, len(pairs_batch[pairs_batch == 0]), axis=0)

    pairs = pairs.astype(np.int64)
    joints_pair = np.concatenate(
        (joints[pairs[:, 0]], joints[pairs[:, 1]], pair_attr[:, :-1]), axis=1)

    # expand_joint_feature
    expand_joint_feature = bone_net['expand_joint_feature']
    if not args.onnx:
        pair_feature = expand_joint_feature.predict({
            'joints_pair': joints_pair,
        })[0]
    else:
        in_x = expand_joint_feature.get_inputs()[0].name
        out_x = expand_joint_feature.get_outputs()[0].name
        pair_feature = expand_joint_feature.run(
            [out_x], {in_x: joints_pair})[0]

    pair_feature = np.concatenate((shape_feature, joint_feature, pair_feature), axis=1)

    # mix_transform
    mix_transform = bone_net['mix_transform']
    if not args.onnx:
        pre_label = mix_transform.predict({
            'pair_feature': pair_feature,
        })[0]
    else:
        in_x = mix_transform.get_inputs()[0].name
        out_x = mix_transform.get_outputs()[0].name
        pre_label = mix_transform.run(
            [out_x], {in_x: pair_feature})[0]

    connect_prob = pre_label
    connect_prob = sigmoid(connect_prob)

    pair_idx = pairs
    prob_matrix = np.zeros((len(joints), len(joints)))
    prob_matrix[pair_idx[:, 0], pair_idx[:, 1]] = connect_prob.squeeze()
    prob_matrix = prob_matrix + prob_matrix.transpose()
    cost_matrix = -np.log(prob_matrix + 1e-10)
    cost_matrix = increase_cost_for_outside_bone(cost_matrix, joints, vox)

    pred_skel = RigInfo()
    parent, key = primMST_symmetry(cost_matrix, root_id, joints)
    for i in range(len(parent)):
        if parent[i] == -1:
            pred_skel.root = TreeNode('root', tuple(joints[i]))
            break
    loadSkel_recur(pred_skel.root, i, None, joints, parent)
    pred_skel.joint_pos = pred_skel.get_joint_dict()

    return pred_skel


def predict_skinning(
        data, pred_skel, skin_net, surface_geodesic,
        subsampling=False, decimation=3000, sampling=1500):
    """
    predict skinning
    :param data: wrapped input data
    :param pred_skel: predicted skeleton
    :param skin_net: network to predict skinning weights
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: predicted rig with skinning weights information
    """
    num_nearest_bone = 5
    bones, bone_names, bone_isleaf = get_bones(pred_skel)
    mesh_v = data['pos']

    print("     calculating volumetric geodesic distance from vertices to bone. This step takes some time...")
    geo_dist = calc_geodesic_matrix(
        bones, mesh_v, surface_geodesic,
        use_sampling=subsampling, decimation=decimation, sampling=sampling)
    input_samples = []  # joint_pos (x, y, z), (bone_id, 1/D)*5
    loss_mask = []
    skin_nn = []
    for v_id in range(len(mesh_v)):
        geo_dist_v = geo_dist[v_id]
        bone_id_near_to_far = np.argsort(geo_dist_v)
        this_sample = []
        this_nn = []
        this_mask = []
        for i in range(num_nearest_bone):
            if i >= len(bones):
                this_sample += bones[bone_id_near_to_far[0]].tolist()
                this_sample.append(1.0 / (geo_dist_v[bone_id_near_to_far[0]] + 1e-10))
                this_sample.append(bone_isleaf[bone_id_near_to_far[0]])
                this_nn.append(0)
                this_mask.append(0)
            else:
                skel_bone_id = bone_id_near_to_far[i]
                this_sample += bones[skel_bone_id].tolist()
                this_sample.append(1.0 / (geo_dist_v[skel_bone_id] + 1e-10))
                this_sample.append(bone_isleaf[skel_bone_id])
                this_nn.append(skel_bone_id)
                this_mask.append(1)
        input_samples.append(np.array(this_sample)[np.newaxis, :])
        skin_nn.append(np.array(this_nn)[np.newaxis, :])
        loss_mask.append(np.array(this_mask)[np.newaxis, :])

    skin_input = np.concatenate(input_samples, axis=0)
    loss_mask = np.concatenate(loss_mask, axis=0)
    skin_nn = np.concatenate(skin_nn, axis=0)
    data['skin_input'] = skin_input

    pos, tpl_edge_index, geo_edge_index, batch = \
        data['pos'], data['tpl_edge_index'], data['geo_edge_index'], data['batch']
    if not args.onnx:
        skin_pred = skin_net.predict({
            'batch': batch, 'pos': pos, 'geo_edge_index': geo_edge_index, 'tpl_edge_index': tpl_edge_index,
            'sample': skin_input.astype(np.float32)
        })[0]
    else:
        in_batch = skin_net.get_inputs()[0].name
        in_pos = skin_net.get_inputs()[1].name
        in_geo_e = skin_net.get_inputs()[2].name
        in_tpl_e = skin_net.get_inputs()[3].name
        in_sample = skin_net.get_inputs()[4].name
        out = skin_net.get_outputs()[0].name
        skin_pred = skin_net.run(
            [out],
            {
                in_batch: batch, in_pos: pos, in_geo_e: geo_edge_index, in_tpl_e: tpl_edge_index,
                in_sample: skin_input.astype(np.float32)
            })[0]

    skin_pred = softmax(skin_pred, axis=1)
    skin_pred = skin_pred * loss_mask

    skin_nn = skin_nn[:, 0:num_nearest_bone]
    skin_pred_full = np.zeros((len(skin_pred), len(bone_names)))
    for v in range(len(skin_pred)):
        for nn_id in range(len(skin_nn[v, :])):
            skin_pred_full[v, skin_nn[v, nn_id]] = skin_pred[v, nn_id]

    print("     filtering skinning prediction")
    tpl_e = tpl_edge_index
    skin_pred_full = post_filter(skin_pred_full, tpl_e, num_ring=1)
    skin_pred_full[skin_pred_full < np.max(skin_pred_full, axis=1, keepdims=True) * 0.35] = 0.0
    skin_pred_full = skin_pred_full / (skin_pred_full.sum(axis=1, keepdims=True) + 1e-10)
    skel_res = assemble_skel_skin(pred_skel, skin_pred_full)

    return skel_res


def recognize_from_obj(
        mesh_filename, net_info,
        downsample_skinning=True, decimation=3000, sampling=1500):
    # prepare input data
    data, vox, surface_geodesic, translation_normalize, scale_normalize = \
        create_single_data(mesh_filename, args.vox_file)

    print("predicting joints")
    data = predict_joints(
        data, vox, net_info['jointNet'], args.threshold, bandwidth=args.bandwidth)

    print("predicting connectivity")
    pred_skel = predict_skeleton(
        data, vox, net_info['rootNet'], net_info['boneNet'])

    print("predicting skinning")
    pred_rig = predict_skinning(
        data, pred_skel, net_info['skinNet'], surface_geodesic,
        subsampling=downsample_skinning, decimation=decimation, sampling=sampling)

    # here we reverse the normalization to the original scale and position
    pred_rig.normalize(scale_normalize, -translation_normalize)

    try:
        img = show_obj_skel(mesh_filename, pred_skel.root)
        cv2.imwrite(args.savepath, img)
    except:
        traceback.print_exc()
        print("Visualization is not supported on headless servers. Please consider other headless rendering methods.")

    # here we use remeshed mesh
    print("Saving result")
    pred_rig.save(mesh_filename.replace('.obj', '_rig.txt'))

    print('Script finished successfully.')


def main():
    # model files check and download
    print('=== JOINETNET ===')
    check_and_download_models(WEIGHT_JOINTNET_PATH, MODEL_JOINTNET_PATH, REMOTE_PATH)
    print('=== ROOTNET (1/8) ===')
    check_and_download_models(WEIGHT_ROOTNET_SE_PATH, MODEL_ROOTNET_SE_PATH, REMOTE_PATH)
    print('=== ROOTNET (2/8) ===')
    check_and_download_models(WEIGHT_ROOTNET_SA1_PATH, MODEL_ROOTNET_SA1_PATH, REMOTE_PATH)
    print('=== ROOTNET (3/8) ===')
    check_and_download_models(WEIGHT_ROOTNET_SA2_PATH, MODEL_ROOTNET_SA2_PATH, REMOTE_PATH)
    print('=== ROOTNET (4/8) ===')
    check_and_download_models(WEIGHT_ROOTNET_SA3_PATH, MODEL_ROOTNET_SA3_PATH, REMOTE_PATH)
    print('=== ROOTNET (5/8) ===')
    check_and_download_models(WEIGHT_ROOTNET_FP3_PATH, MODEL_ROOTNET_FP3_PATH, REMOTE_PATH)
    print('=== ROOTNET (6/8) ===')
    check_and_download_models(WEIGHT_ROOTNET_FP2_PATH, MODEL_ROOTNET_FP2_PATH, REMOTE_PATH)
    print('=== ROOTNET (7/8) ===')
    check_and_download_models(WEIGHT_ROOTNET_FP1_PATH, MODEL_ROOTNET_FP1_PATH, REMOTE_PATH)
    print('=== ROOTNET (8/8) ===')
    check_and_download_models(WEIGHT_ROOTNET_BL_PATH, MODEL_ROOTNET_BL_PATH, REMOTE_PATH)
    print('=== BONENET (1/6) ===')
    check_and_download_models(WEIGHT_BONENET_SA1_PATH, MODEL_BONENET_SA1_PATH, REMOTE_PATH)
    print('=== BONENET (2/6) ===')
    check_and_download_models(WEIGHT_BONENET_SA2_PATH, MODEL_BONENET_SA2_PATH, REMOTE_PATH)
    print('=== BONENET (3/6) ===')
    check_and_download_models(WEIGHT_BONENET_SA3_PATH, MODEL_BONENET_SA3_PATH, REMOTE_PATH)
    print('=== BONENET (4/6) ===')
    check_and_download_models(WEIGHT_BONENET_SE_PATH, MODEL_BONENET_SE_PATH, REMOTE_PATH)
    print('=== BONENET (5/6) ===')
    check_and_download_models(WEIGHT_BONENET_EF_PATH, MODEL_BONENET_EF_PATH, REMOTE_PATH)
    print('=== BONENET (6/6) ===')
    check_and_download_models(WEIGHT_BONENET_MT_PATH, MODEL_BONENET_MT_PATH, REMOTE_PATH)
    print('=== SKINNET ===')
    check_and_download_models(WEIGHT_SKINNET_PATH, MODEL_SKINNET_PATH, REMOTE_PATH)

    # load model
    net_info = {}
    if not args.onnx:
        env_id = args.env_id

        net_info['jointNet'] = ailia.Net(MODEL_JOINTNET_PATH, WEIGHT_JOINTNET_PATH, env_id=env_id)
        net_info['rootNet'] = {
            'shape_encoder': ailia.Net(MODEL_ROOTNET_SE_PATH, WEIGHT_ROOTNET_SE_PATH, env_id=env_id),
            'sa1_module': ailia.Net(MODEL_ROOTNET_SA1_PATH, WEIGHT_ROOTNET_SA1_PATH, env_id=env_id),
            'sa2_module': ailia.Net(MODEL_ROOTNET_SA2_PATH, WEIGHT_ROOTNET_SA2_PATH, env_id=env_id),
            'sa3_module': ailia.Net(MODEL_ROOTNET_SA3_PATH, WEIGHT_ROOTNET_SA3_PATH, env_id=env_id),
            'fp3_module': ailia.Net(MODEL_ROOTNET_FP3_PATH, WEIGHT_ROOTNET_FP3_PATH, env_id=env_id),
            'fp2_module': ailia.Net(MODEL_ROOTNET_FP2_PATH, WEIGHT_ROOTNET_FP2_PATH, env_id=env_id),
            'fp1_module': ailia.Net(MODEL_ROOTNET_FP1_PATH, WEIGHT_ROOTNET_FP1_PATH, env_id=env_id),
            'back_layers': ailia.Net(MODEL_ROOTNET_BL_PATH, WEIGHT_ROOTNET_BL_PATH, env_id=env_id),
        }
        net_info['boneNet'] = {
            'sa1_module': ailia.Net(MODEL_BONENET_SA1_PATH, WEIGHT_BONENET_SA1_PATH, env_id=env_id),
            'sa2_module': ailia.Net(MODEL_BONENET_SA2_PATH, WEIGHT_BONENET_SA2_PATH, env_id=env_id),
            'sa3_module': ailia.Net(MODEL_BONENET_SA3_PATH, WEIGHT_BONENET_SA3_PATH, env_id=env_id),
            'shape_encoder': ailia.Net(MODEL_BONENET_SE_PATH, WEIGHT_BONENET_SE_PATH, env_id=env_id),
            'expand_joint_feature': ailia.Net(MODEL_BONENET_EF_PATH, WEIGHT_BONENET_EF_PATH, env_id=env_id),
            'mix_transform': ailia.Net(MODEL_BONENET_MT_PATH, WEIGHT_BONENET_MT_PATH, env_id=env_id),
        }
        net_info['skinNet'] = ailia.Net(MODEL_SKINNET_PATH, WEIGHT_SKINNET_PATH, env_id=env_id),
    else:
        import onnxruntime
        net_info['jointNet'] = onnxruntime.InferenceSession(WEIGHT_JOINTNET_PATH)
        net_info['rootNet'] = {
            'shape_encoder': onnxruntime.InferenceSession(WEIGHT_ROOTNET_SE_PATH),
            'sa1_module': onnxruntime.InferenceSession(WEIGHT_ROOTNET_SA1_PATH),
            'sa2_module': onnxruntime.InferenceSession(WEIGHT_ROOTNET_SA2_PATH),
            'sa3_module': onnxruntime.InferenceSession(WEIGHT_ROOTNET_SA3_PATH),
            'fp3_module': onnxruntime.InferenceSession(WEIGHT_ROOTNET_FP3_PATH),
            'fp2_module': onnxruntime.InferenceSession(WEIGHT_ROOTNET_FP2_PATH),
            'fp1_module': onnxruntime.InferenceSession(WEIGHT_ROOTNET_FP1_PATH),
            'back_layers': onnxruntime.InferenceSession(WEIGHT_ROOTNET_BL_PATH),
        }
        net_info['boneNet'] = {
            'sa1_module': onnxruntime.InferenceSession(WEIGHT_BONENET_SA1_PATH),
            'sa2_module': onnxruntime.InferenceSession(WEIGHT_BONENET_SA2_PATH),
            'sa3_module': onnxruntime.InferenceSession(WEIGHT_BONENET_SA3_PATH),
            'shape_encoder': onnxruntime.InferenceSession(WEIGHT_BONENET_SE_PATH),
            'expand_joint_feature': onnxruntime.InferenceSession(WEIGHT_BONENET_EF_PATH),
            'mix_transform': onnxruntime.InferenceSession(WEIGHT_BONENET_MT_PATH),
        }
        net_info['skinNet'] = onnxruntime.InferenceSession(WEIGHT_SKINNET_PATH)

    recognize_from_obj(args.input, net_info)


if __name__ == '__main__':
    main()
