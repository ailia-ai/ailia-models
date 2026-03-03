import os

import numpy as np
# import scipy.misc
from PIL import Image

# logger
from logging import getLogger  # noqa: E402

__all__ = [
    'get_raw_bodies_data',
    'get_raw_denoised_data',
    'seq_translation',
    'align_frames',
    'torgb',
]

logger = getLogger(__name__)


def get_raw_bodies_data(skes_path):
    """
    Get raw bodies data from a skeleton sequence.

    Each body's data is a dict that contains the following keys:
      - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
      - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
      - interval: a list which stores the frame indices of this body.
      - motion: motion amount (only for the sequence with 2 or more bodyIDs).

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """
    logger.info("get_raw_bodies_data")

    ske_file = os.path.basename(skes_path)
    ske_name, _ = os.path.splitext(ske_file)

    # Read all data from .skeleton file into a list (in string format)
    with open(skes_path, 'r') as fr:
        str_data = fr.readlines()

    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1  # 0-based index
    current_line = 1

    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1

        if num_bodies == 0:  # no data in this frame, drop it
            frames_drop.append(f)  # 0-based index
            continue

        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

        for b in range(num_bodies):
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
            current_line += 1

            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                current_line += 1

            if bodyID not in bodies_data:  # Add a new body's data
                body_data = dict()
                body_data['joints'] = joints[b]  # ndarray: (25, 3)
                body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2)
                body_data['interval'] = [valid_frames]  # the index of the first frame
            else:  # Update an already existed body's data
                body_data = bodies_data[bodyID]
                # Stack each body's data of each frame along the frame order
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index

            bodies_data[bodyID] = body_data  # Update bodies_data

    num_frames_drop = len(frames_drop)
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) is missing or lost' % num_frames

    if num_frames_drop > 0:
        logger.info('{} frames missed: {}\n'.format(
            num_frames_drop, frames_drop))

    # Calculate motion (only for the sequence with 2 or more bodyIDs)
    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def denoising_by_length(ske_name, bodies_data, noise_len_thres=11):
    """
    Denoising data based on the frame length for each bodyID.
    Filter out the bodyID which length is less or equal than the predefined threshold.

    """
    noise_info = str()
    new_bodies_data = bodies_data.copy()
    for (bodyID, body_data) in new_bodies_data.items():
        length = len(body_data['interval'])
        if length <= noise_len_thres:
            noise_info += 'Filter out: %s, %d (length).\n' % (bodyID, length)
            logger.info('{}\t{}\t{:.6f}\t{:^6d}'.format(
                ske_name, bodyID, body_data['motion'], length))
            del bodies_data[bodyID]
    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info


def get_valid_frames_by_spread(points, noise_spr_thres1=0.8):
    """
    Find the valid (or reasonable) frames (index) based on the spread of X and Y.

    :param points: joints or colors
    """
    num_frames = points.shape[0]
    valid_frames = []
    for i in range(num_frames):
        x = points[i, :, 0]
        y = points[i, :, 1]
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):  # 0.8
            valid_frames.append(i)
    return valid_frames


def denoising_by_spread(ske_name, bodies_data, noise_spr_thres2=0.69754):
    """
    Denoising data based on the spread of Y value and X value.
    Filter out the bodyID which the ratio of noisy frames is higher than the predefined
    threshold.

    bodies_data: contains at least 2 bodyIDs
    """
    noise_info = str()
    denoised_by_spr = False  # mark if this sequence has been processed by spread.

    new_bodies_data = bodies_data.copy()
    # for (bodyID, body_data) in bodies_data.items():
    for (bodyID, body_data) in new_bodies_data.items():
        if len(bodies_data) == 1:
            break

        valid_frames = get_valid_frames_by_spread(body_data['joints'].reshape(-1, 25, 3))
        num_frames = len(body_data['interval'])
        num_noise = num_frames - len(valid_frames)
        if num_noise == 0:
            continue

        ratio = num_noise / float(num_frames)
        motion = body_data['motion']
        if ratio >= noise_spr_thres2:  # 0.69754
            del bodies_data[bodyID]
            denoised_by_spr = True
            noise_info += 'Filter out: %s (spread rate >= %.2f).\n' % (bodyID, noise_spr_thres2)
            logger.info('%s\t%s\t%.6f\t%.6f' % (ske_name, bodyID, motion, ratio))
        else:  # Update motion
            joints = body_data['joints'].reshape(-1, 25, 3)[valid_frames]
            body_data['motion'] = min(motion, np.sum(np.var(joints.reshape(-1, 3), axis=0)))
            noise_info += '%s: motion %.6f -> %.6f\n' % (bodyID, motion, body_data['motion'])
            # TODO: Consider removing noisy frames for each bodyID

    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info, denoised_by_spr


def denoising_bodies_data(bodies_data):
    """
    Denoising data based on some heuristic methods, not necessarily correct for all samples.

    Return:
      denoised_bodies_data (list): tuple: (bodyID, body_data).
    """
    ske_name = bodies_data['name']
    bodies_data = bodies_data['data']

    # Step 1: Denoising based on frame length.
    bodies_data, noise_info_len = denoising_by_length(ske_name, bodies_data)

    if len(bodies_data) == 1:  # only has one bodyID left after step 1
        return bodies_data.items(), noise_info_len

    # Step 2: Denoising based on spread.
    bodies_data, noise_info_spr, denoised_by_spr = denoising_by_spread(ske_name, bodies_data)

    if len(bodies_data) == 1:
        return bodies_data.items(), noise_info_len + noise_info_spr

    bodies_motion = dict()  # get body motion
    for (bodyID, body_data) in bodies_data.items():
        bodies_motion[bodyID] = body_data['motion']
    # Sort bodies based on the motion
    # bodies_motion = sorted(bodies_motion.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
    bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
    denoised_bodies_data = list()
    for (bodyID, _) in bodies_motion:
        denoised_bodies_data.append((bodyID, bodies_data[bodyID]))

    return denoised_bodies_data, noise_info_len + noise_info_spr

    # TODO: Consider denoising further by integrating motion method

    # if denoised_by_spr:  # this sequence has been denoised by spread
    #     bodies_motion = sorted(bodies_motion.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    #     denoised_bodies_data = list()
    #     for (bodyID, _) in bodies_motion:
    #         denoised_bodies_data.append((bodyID, bodies_data[bodyID]))
    #     return denoised_bodies_data, noise_info

    # Step 3: Denoising based on motion
    # bodies_data, noise_info = denoising_by_motion(ske_name, bodies_data, bodies_motion)

    # return bodies_data, noise_info


def get_one_actor_points(body_data, num_frames):
    """
    Get joints and colors for only one actor.
    For joints, each frame contains 75 X-Y-Z coordinates.
    For colors, each frame contains 25 x 2 (X, Y) coordinates.
    """
    joints = np.zeros((num_frames, 75), dtype=np.float32)
    colors = np.ones((num_frames, 1, 25, 2), dtype=np.float32) * np.nan
    start, end = body_data['interval'][0], body_data['interval'][-1]
    joints[start:end + 1] = body_data['joints'].reshape(-1, 75)
    colors[start:end + 1, 0] = body_data['colors']

    return joints, colors


def remove_missing_frames(ske_name, joints, colors):
    """
    Cut off missing frames which all joints positions are 0s

    For the sequence with 2 actors' data, also record the number of missing frames for
    actor1 and actor2, respectively (for debug).
    """
    num_frames = joints.shape[0]
    num_bodies = colors.shape[1]  # 1 or 2

    if num_bodies == 2:  # DEBUG
        missing_indices_1 = np.where(joints[:, :75].sum(axis=1) == 0)[0]
        missing_indices_2 = np.where(joints[:, 75:].sum(axis=1) == 0)[0]
        cnt1 = len(missing_indices_1)
        cnt2 = len(missing_indices_2)

        start = 1 if 0 in missing_indices_1 else 0
        end = 1 if num_frames - 1 in missing_indices_1 else 0
        if max(cnt1, cnt2) > 0:
            if cnt1 > cnt2:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}\t{:^5d}\t{:^3d}'.format(ske_name, num_frames,
                                                                            cnt1, cnt2, start, end)
                logger.info(info)
            else:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}'.format(ske_name, num_frames, cnt1, cnt2)
                logger.info(info)

    # Find valid frame indices that the data is not missing or lost
    # For two-subjects action, this means both data of actor1 and actor2 is missing.
    valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # 0-based index
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]
    num_missing = len(missing_indices)

    if num_missing > 0:  # Update joints and colors
        joints = joints[valid_indices]
        colors[missing_indices] = np.nan
        global missing_count
        missing_count += 1
        logger.info('{}\t{:^10d}\t{:^11d}'.format(ske_name, num_frames, num_missing))

    return joints, colors


def get_bodies_info(bodies_data):
    bodies_info = '{:^17}\t{}\t{:^8}\n'.format('bodyID', 'Interval', 'Motion')
    for (bodyID, body_data) in bodies_data.items():
        start, end = body_data['interval'][0], body_data['interval'][-1]
        bodies_info += '{}\t{:^8}\t{:f}\n'.format(
            bodyID, str([start, end]), body_data['motion'])

    return bodies_info + '\n'


def get_two_actors_points(bodies_data):
    """
    Get the first and second actor's joints positions and colors locations.

    # Arguments:
        bodies_data (dict): 3 key-value pairs: 'name', 'data', 'num_frames'.
        bodies_data['data'] is also a dict, while the key is bodyID, the value is
        the corresponding body_data which is also a dict with 4 keys:
          - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
          - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
          - interval: a list which records the frame indices.
          - motion: motion amount

    # Return:
        joints, colors.
    """
    ske_name = bodies_data['name']
    label = int(ske_name[-2:])
    num_frames = bodies_data['num_frames']
    bodies_info = get_bodies_info(bodies_data['data'])

    bodies_data, noise_info = denoising_bodies_data(bodies_data)  # Denoising data
    bodies_info += noise_info

    bodies_data = list(bodies_data)
    if len(bodies_data) == 1:  # Only left one actor after denoising
        if label >= 50:  # DEBUG: Denoising failed for two-subjects action
            logger.info(ske_name)

        bodyID, body_data = bodies_data[0]
        joints, colors = get_one_actor_points(body_data, num_frames)
        bodies_info += 'Main actor: %s' % bodyID
    else:
        if label < 50:  # DEBUG: Denoising failed for one-subject action
            logger.info(ske_name)

        joints = np.zeros((num_frames, 150), dtype=np.float32)
        colors = np.ones((num_frames, 2, 25, 2), dtype=np.float32) * np.nan

        bodyID, actor1 = bodies_data[0]  # the 1st actor with largest motion
        start1, end1 = actor1['interval'][0], actor1['interval'][-1]
        joints[start1:end1 + 1, :75] = actor1['joints'].reshape(-1, 75)
        colors[start1:end1 + 1, 0] = actor1['colors']
        actor1_info = '{:^17}\t{}\t{:^8}\n'.format('Actor1', 'Interval', 'Motion') + \
                      '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start1, end1]), actor1['motion'])
        del bodies_data[0]

        actor2_info = '{:^17}\t{}\t{:^8}\n'.format('Actor2', 'Interval', 'Motion')
        start2, end2 = [0, 0]  # initial interval for actor2 (virtual)

        while len(bodies_data) > 0:
            bodyID, actor = bodies_data[0]
            start, end = actor['interval'][0], actor['interval'][-1]
            if min(end1, end) - max(start1, start) <= 0:  # no overlap with actor1
                joints[start:end + 1, :75] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 0] = actor['colors']
                actor1_info += '{}\t{:^8}\t{:f}\n'.format(
                    bodyID, str([start, end]), actor['motion'])
                # Update the interval of actor1
                start1 = min(start, start1)
                end1 = max(end, end1)
            elif min(end2, end) - max(start2, start) <= 0:  # no overlap with actor2
                joints[start:end + 1, 75:] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 1] = actor['colors']
                actor2_info += '{}\t{:^8}\t{:f}\n'.format(
                    bodyID, str([start, end]), actor['motion'])
                # Update the interval of actor2
                start2 = min(start, start2)
                end2 = max(end, end2)
            del bodies_data[0]

        bodies_info += ('\n' + actor1_info + '\n' + actor2_info)

    logger.info(bodies_info)

    return joints, colors


def get_raw_denoised_data(bodies_data):
    """
    Get denoised data (joints positions and color locations) from raw skeleton sequences.

    For each frame of a skeleton sequence, an actor's 3D positions of 25 joints represented
    by an 2D array (shape: 25 x 3) is reshaped into a 75-dim vector by concatenating each
    3-dim (x, y, z) coordinates along the row dimension in joint order. Each frame contains
    two actor's joints positions constituting a 150-dim vector. If there is only one actor,
    then the last 75 values are filled with zeros. Otherwise, select the main actor and the
    second actor based on the motion amount. Each 150-dim vector as a row vector is put into
    a 2D numpy array where the number of rows equals the number of valid frames. All such
    2D arrays are put into a list and finally the list is serialized into a cPickle file.

    For the skeleton sequence which contains two or more actors (mostly corresponds to the
    last 11 classes), the filename and actors' information are recorded into log files.
    For better understanding, also generate RGB+skeleton videos for visualization.
    """

    ske_name = bodies_data['name']
    num_bodies = len(bodies_data['data'])

    if num_bodies == 1:  # only 1 actor
        num_frames = bodies_data['num_frames']
        body_data = list(bodies_data['data'].values())[0]
        joints, colors = get_one_actor_points(body_data, num_frames)
    else:  # more than 1 actor, select two main actors
        joints, colors = get_two_actors_points(bodies_data)
        # Remove missing frames
        joints, colors = remove_missing_frames(ske_name, joints, colors)
        num_frames = joints.shape[0]  # Update
        # Visualize selected actors' skeletons on RGB videos.

    return num_frames, joints, colors


def seq_translation(ske_joints):
    num_frames = ske_joints.shape[0]
    num_bodies = 1 if ske_joints.shape[1] == 75 else 2
    if num_bodies == 2:
        missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
        missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
        cnt1 = len(missing_frames_1)
        cnt2 = len(missing_frames_2)

    i = 0  # get the "real" first frame of actor1
    while i < num_frames:
        if np.any(ske_joints[i, :75] != 0):
            break
        i += 1

    origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

    for f in range(num_frames):
        if num_bodies == 1:
            ske_joints[f] -= np.tile(origin, 25)
        else:  # for 2 actors
            ske_joints[f] -= np.tile(origin, 50)

    if (num_bodies == 2) and (cnt1 > 0):
        ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

    if (num_bodies == 2) and (cnt2 > 0):
        ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

    return ske_joints


def align_frames(ske_joints, frames_cnt=300):
    """
    Align all sequences with the same frame length.

    """
    aligned_skes_joints = np.zeros((frames_cnt, 150), dtype=np.float32)

    num_frames = ske_joints.shape[0]
    num_bodies = 1 if ske_joints.shape[1] == 75 else 2
    if num_bodies == 1:
        aligned_skes_joints[:num_frames] = np.hstack(
            (ske_joints, np.zeros_like(ske_joints))
        )
    else:
        aligned_skes_joints[:num_frames] = ske_joints

    return aligned_skes_joints


def bytescale(data, high=255, low=0):
    cmin = data.min()
    cmax = data.max()

    cscale = cmax - cmin

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(data, high=255, low=0):
    shape = list(data.shape)

    bytedata = bytescale(data, high=high, low=low)
    strdata = bytedata.tobytes()
    shape = (shape[1], shape[0])

    image = Image.frombytes('RGB', shape, strdata)
    return image


def _center(rgb):
    rgb[:, :, 0] -= 110
    rgb[:, :, 1] -= 110
    rgb[:, :, 2] -= 110
    return rgb


def torgb(ske_joints, max_val, min_val):
    rgb = []
    maxmin = list()

    for ske_joint in ske_joints:
        zero_row = []

        for i in range(len(ske_joint)):
            if (ske_joint[i, :] == np.zeros((1, 150))).all():
                zero_row.append(i)
        ske_joint = np.delete(ske_joint, zero_row, axis=0)
        if (ske_joint[:, 0:75] == np.zeros((ske_joint.shape[0], 75))).all():
            ske_joint = np.delete(ske_joint, range(75), axis=1)
        elif (ske_joint[:, 75:150] == np.zeros((ske_joint.shape[0], 75))).all():
            ske_joint = np.delete(ske_joint, range(75, 150), axis=1)

        #### original rescale to 0-255
        ske_joint = 255 * (ske_joint - min_val) / (max_val - min_val)
        rgb_ske = np.reshape(ske_joint, (ske_joint.shape[0], ske_joint.shape[1] // 3, 3))
        # rgb_ske = scipy.misc.imresize(rgb_ske, (224, 224)).astype(np.float32)
        rgb_ske = np.array(toimage(rgb_ske).resize(
            (224, 224), resample=Image.BILINEAR)).astype(np.float32)
        rgb_ske = _center(rgb_ske)
        rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
        rgb_ske = np.transpose(rgb_ske, [2, 1, 0])
        rgb.append(rgb_ske)
        maxmin.append([max_val, min_val])

    return rgb, maxmin
