# PaDiM GUI launcher

import os
import cv2
import numpy
import shutil
import sys
import glob
import ailia

from PIL import Image, ImageTk

sys.path.append('../../util')
from padim_utils import *
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# for macOS, please install "brew install python-tk@3.9"
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog

import log_init
from logging import getLogger  # noqa: E402

from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)

parser = get_base_parser('PaDiM GUI', None, None)
args = update_parser(parser)

# ======================
# Global settings
# ======================

input_index = 0
output_index = 0
result_index = 0
model_index = 0
slider_index = 50


REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/padim/'

train_folder = None
test_folder = None
test_type = "folder"
test_roi = None
score_cache = {}
EnableOptimization = True
save_format="pkl"
BENCHMARK = False
selected_device = "cpu"
if EnableOptimization:
    import torch
    
    device = torch.device("cpu")
    weights_torch=gaussian_kernel1d_torch(4, 0, int(4.0*float(4)+0.5), device).unsqueeze(0).unsqueeze(0).expand(1, 1, 33)
    

    logger.info("Torch device : " + str(device))

# ======================
# List box cursor changed
# ======================

def input_changed(event):
    global input_index
    selection = event.widget.curselection()
    if selection:
        input_index = selection[0]
    else:
        input_index = 0   
    load_detail(train_list[input_index], True)

def output_changed(event):
    global output_index
    selection = event.widget.curselection()
    if selection:
        output_index = selection[0]
    else:
        output_index = 0   
    load_detail(test_list[output_index], True)

def result_changed(event):
    global result_index
    selection = event.widget.curselection()
    if selection:
        result_index = selection[0]
    else:
        result_index = 0   
    load_detail(result_list[result_index], False)

def model_changed(event):
    global model_index
    selection = event.widget.curselection()
    if selection:
        model_index = selection[0]
    else:
        model_index = 0   

def slider_changed(event):
    global scale, slider_index
    slider_index = scale.get()

def enable_optimization(event):
    global EnableOptimization
    selection = event.widget.curselection()
    if selection:
        selected_index = selection[0]
        selected_value = event.widget.get(selected_index)
        EnableOptimization = (selected_value == "True")
    else:
        EnableOptimization = False
    logger.info(f"EnableOptimization set to: {EnableOptimization}")

def save_type_select(event):
    global save_format
    selection = event.widget.curselection()
    if selection:
        selected_index = selection[0]
        selected_value = event.widget.get(selected_index)
        save_format = selected_value
    else:
        save_format = "pkl"
    logger.info(f"Selected format set to: {save_format}")

def enable_benchmark(event):
    global BENCHMARK
    selection = event.widget.curselection()
    if selection:
        selected_index = selection[0]
        selected_value = event.widget.get(selected_index)
        BENCHMARK = (selected_value == "True")
    else:
        BENCHMARK = False
    logger.info(f"BENCHMARK set to: {BENCHMARK}")

def select_device(event):
    global device, weights_torch
    selection = event.widget.curselection()
    if selection:
        selected_index = selection[0]
        selected_value = event.widget.get(selected_index)
        device = torch.device(selected_value)
        weights_torch = gaussian_kernel1d_torch(4, 0, int(4.0*float(4)+0.5), device).unsqueeze(0).unsqueeze(0).expand(1, 1, 33)
    
        if selected_value=="cuda:0":
            torch.cuda.set_per_process_memory_fraction(fraction=0.5, device="cuda:0")
    else:
        device = torch.device("cpu")

    
    logger.info(f"Device set to: {device}")

# ======================
# List box double click
# ======================

def open_file_by_os(filepath):
    import subprocess, os, platform
    if platform.system() == 'Darwin':       # macOS
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':    # Windows
        os.startfile(filepath)
    else:                                   # linux variants
        subprocess.call(('xdg-open', filepath))

def input_double_click(event):
    global input_index
    selection = event.widget.curselection()
    if selection:
        input_index = selection[0]
    else:
        input_index = 0   
    open_file_by_os(train_list[input_index])

def output_double_click(event):
    global output_index
    selection = event.widget.curselection()
    if selection:
        output_index = selection[0]
    else:
        output_index = 0   
    open_file_by_os(test_list[output_index])

def result_double_click(event):
    global result_index
    selection = event.widget.curselection()
    if selection:
        result_index = selection[0]
    else:
        result_index = 0
    open_file_by_os(result_list[result_index])

# ======================
# Change file
# ======================

CANVAS_W = 480
CANVAS_H = 160

def create_photo_image(path,w=CANVAS_W,h=CANVAS_H):
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        capture = cv2.VideoCapture(path)
        ret, image_bgr = capture.read()
        capture.release()
    #image_bgr = cv2.resize(image_bgr,(w,h))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
    image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
    image_pil.thumbnail((w,h), Image.LANCZOS)
    image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換
    return image_tk

def load_canvas_image(path, is_train):
    global canvas, canvas_item, image_tk
    if is_train:
        image_tk = create_photo_image(path,w=CANVAS_H,h=CANVAS_H)
    else:
        image_tk = create_photo_image(path)
    if canvas_item == None:
        canvas_item = canvas.create_image(0, 0, image=image_tk, anchor=tk.NW)
    else:
        canvas.itemconfig(canvas_item,image=image_tk)

def load_roi_image(path):
    global canvas_roi, canvas_roi_item, image_tk_roi
    image_tk_roi = create_photo_image(path,w=CANVAS_H,h=CANVAS_H)
    if canvas_roi_item == None:
        canvas_roi_item = canvas_roi.create_image(0, 0, image=image_tk_roi, anchor=tk.NW)
    else:
        canvas_roi.itemconfig(canvas_roi_item,image=image_tk_roi)

def load_detail(image_path, is_train):
    image_exist = False
    for ext in [".jpg",".png"]:
        if os.path.exists(image_path):
            load_canvas_image(image_path, is_train)
            image_exist = True
            break


# ======================
# Run model
# ======================

def get_keep_aspect():
    global valueKeepAspect
    return valueKeepAspect.get()

def get_image_resize():
    global valueCenterCrop
    image_resize = get_image_crop_size()
    if valueCenterCrop.get():
        return get_image_crop_size() + (256 - 224)
    return image_resize

def get_image_crop_size():
    global model_index
    return get_model_resolution_list()[model_index]

def get_model():
    global model_index
    return get_model_list()[model_index]

def get_model_id():
    global model_index
    return get_model_id_list()[model_index]

def train_button_clicked():
    global train_folder
    print("begin training")

    # model files check and download
    weight_path, model_path, params = get_params(get_model_id())
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # create net instance
    net = ailia.Net(model_path, weight_path, env_id=args.env_id)

    # training
    batch_size = 32
    train_dir = train_folder
    if train_dir == "camera":
        global train_list, input_index
        train_dir = train_list[input_index].split(":")[1]
    aug = False
    aug_num = 0
    seed = 1024
    train_outputs=training_optimized(net, params, get_image_resize(), get_image_crop_size(), get_keep_aspect(), batch_size, train_dir, aug, aug_num, seed, logger)
    if  not EnableOptimization:
        if save_format == "pkl" :
            train_feat_file = "train.pkl"
            logger.info('Saving train set feature to: %s ...' % train_feat_file)
            with open(train_feat_file, 'wb') as f:
                pickle.dump(train_outputs, f)
            logger.info('Saved.')
        elif save_format == "npy" :
            
            for i, output in enumerate(train_outputs):
                    train_feat_file = "train_output_"+str(i)+".npy"
                    logger.info('Saving train set feature to: %s ...' % train_feat_file)
                    np.save(f"train_output_{i}.npy", output) 
            logger.info('Saved.')  
        elif save_format == "pt":
            train_feat_file = "train.pt"
            logger.info('Saving train set feature to: %s ...' % train_feat_file)
            torch.save(train_outputs, train_feat_file)
            logger.info('Saved.')

    else:
        if save_format=="npy":
            for i, output in enumerate(train_outputs):
                    train_feat_file = "train_output_"+str(i)+".npy"
                    logger.info('Saving train set feature to: %s ...' % train_feat_file)
                    np.save(f"train_output_{i}.npy", output) 
            logger.info('Saved.')   
    
        train_outputs=[torch.from_numpy(train_outputs[0]).float().to(device), train_outputs[1], 
                        torch.from_numpy(train_outputs[2]).float().to(device), train_outputs[3] ] 
        if save_format == "pkl" :
            train_feat_file = "trainOptimized.pkl"
            logger.info('saving train set feature to: %s ...' % train_feat_file)
            with open(train_feat_file, 'wb') as f:
                pickle.dump(train_outputs, f)
                logger.info('Saved.')
        elif save_format == "pt":
            train_feat_file = "trainOptimized.pt"
            logger.info('Saving train set feature to: %s ...' % train_feat_file)
            torch.save(train_outputs, train_feat_file)
            logger.info('Saved.')
        
    
       

    global score_cache
    score_cache = {}

def test_button_clicked():
    global score_cache
    global valueKeepAspect, valueCenterCrop
    print("begin test")

    if "keep_aspect" in score_cache:
        if score_cache["keep_aspect"] != get_keep_aspect() or score_cache["image_resize"] != get_image_resize() or score_cache["model"] != get_model():
            score_cache = {}
    score_cache["keep_aspect"] = get_keep_aspect()
    score_cache["image_resize"] = get_image_resize()
    score_cache["model"] = get_model()

    # model files check and download
    weight_path, model_path, params = get_params(get_model_id())
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # create net instance
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(model_path, weight_path, env_id=env_id)
    if _check_file_exists(save_format):
        if EnableOptimization:
            train_feat_file = "trainOptimized."+save_format
            
            if save_format== "pkl":
                logger.info(f"Loading {train_feat_file}")
                with open(train_feat_file, 'rb') as f:
                    train_outputs = pickle.load(f)
                train_outputs=[train_outputs[0].to(device), train_outputs[1], 
                                train_outputs[2].to(device), train_outputs[3] ] 
            elif save_format == "npy":
                train_outputs = []
                i = 0
                while True:
                    try:
                        logger.info(f"Loading train_output_{i}.npy")
                        train_outputs.append(np.load(f"train_output_{i}.npy", allow_pickle=True))
                        i += 1
                    except FileNotFoundError:
                        break  # Stop when there are no more files to load  
                train_outputs=[torch.from_numpy(train_outputs[0]).float().to(device), train_outputs[1], 
                                torch.from_numpy(train_outputs[2]).float().to(device), train_outputs[3] ] 
            elif save_format == "pt":
                logger.info(f"Loading {train_feat_file}")
                train_outputs = torch.load(train_feat_file)
        else:
            train_feat_file = "train."+save_format
            if save_format == "pkl":
                logger.info(f"Loading {train_feat_file}")
                with open(train_feat_file, 'rb') as f:
                    train_outputs = pickle.load(f)
            elif save_format == "npy":
                train_outputs = []
                i = 0
                while True:
                    try:
                        logger.info(f"Loading train_output_{i}.npy")
                        train_outputs.append(np.load(f"train_output_{i}.npy", allow_pickle=True))
                        i += 1
                    except FileNotFoundError:
                        break  # Stop when there are no more files to load
                
            elif save_format == "pt":
                logger.info(f"Loading {train_feat_file}")
                train_outputs = torch.load(train_feat_file)
                train_outputs_numpy = []
                if train_outputs[0] is torch.Tensor:
                    for item in train_outputs:
                        if isinstance(item, torch.Tensor):
                            train_outputs_numpy.append(item.cpu().numpy())  # Move to CPU and convert to NumPy
                        else:
                            train_outputs_numpy.append(item)

    threshold = slider_index / 100.0

    if test_type == "folder":
        test_from_folder(net, params, train_outputs, threshold)
    else:
        test_from_video(net, params, train_outputs, threshold)

def _check_file_exists(save_format):
    if save_format=="npy":
        filename="train_output_0.npy" 
    elif EnableOptimization:
        filename="trainOptimized."+save_format
    else:
        filename="train."+save_format
    if not os.path.isfile(filename):
        logger.info(f"File {filename} does not exist. Unable to load the model")

    return os.path.isfile(filename)

def test_from_folder(net, params, train_outputs, threshold):
    # file loop
    test_imgs = []

    global test_roi
    if test_roi:
        roi_img = load_image(test_roi)
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGRA2RGB)
        roi_img = preprocess(roi_img, get_image_resize(), keep_aspect=get_keep_aspect(), crop_size=get_image_crop_size(), mask=True)
    else:
        roi_img = None
    
    score_map = []
    for i_img in range(0, len(test_list)):
        logger.info('from (%s) ' % (test_list[i_img]))

        image_path = test_list[i_img]
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img, get_image_resize(), keep_aspect=get_keep_aspect(), crop_size=get_image_crop_size())

        test_imgs.append(img[0])
        if image_path in score_cache:
            dist_tmp = score_cache[image_path].copy()
        else:
            if BENCHMARK:
                import time
                logger.info('BENCHMARK mode')
                total_time = 0
                if EnableOptimization:
                    for i in range(6):
                        start = int(round(time.time() * 1000))
                        dist_tmp = infer_optimized(net, params, train_outputs, img, get_image_crop_size(), device, logger, weights_torch)
                        end = int(round(time.time() * 1000))
                        logger.info(f'\tailia processing time {end - start} ms')
                        if i != 0:
                            total_time = total_time + (end - start)
                    logger.info(f'\taverage time {total_time / 5} ms')
                else:
                    for i in range(6):
                        start = int(round(time.time() * 1000))
                        dist_tmp = infer(net, params, train_outputs, img, get_image_crop_size())
                        end = int(round(time.time() * 1000))
                        logger.info(f'\tailia processing time {end - start} ms')
                        if i != 0:
                            total_time = total_time + (end - start)
                    logger.info(f'\taverage time {total_time / 5} ms')

            else:
                if EnableOptimization:
                    dist_tmp = infer_optimized(net, params, train_outputs, img, get_image_crop_size(), device, logger, weights_torch)
                    score_cache[image_path] = dist_tmp
                else:
                    dist_tmp = infer(net, params, train_outputs, img, get_image_crop_size())
                    score_cache[image_path] = dist_tmp.copy()
        score_map.append(dist_tmp)
    if EnableOptimization:
        scores = normalize_scores_torch(score_map, get_image_crop_size()).squeeze(0).cpu().numpy()
        anormal_scores = calculate_anormal_scores_torch(score_map, get_image_crop_size())
    else:
        scores = normalize_scores(score_map, get_image_crop_size(), roi_img)
        anormal_scores = calculate_anormal_scores(score_map, get_image_crop_size())

    # Plot gt image
    os.makedirs("result", exist_ok=True)
    global result_list, listsResult, ListboxResult
    result_list = []
    scores_numpy=np.zeros(scores.shape)
    for i in range(0, len(test_imgs)):
        img = denormalization(test_imgs[i])
        heat_map, mask, vis_img = visualize(img, scores[i], threshold)
        """
        if EnableOptimization:
            scores_numpy[i] = scores[i].squeeze(0).cpu().numpy()
            heat_map, mask, vis_img = visualize(img, scores[i].squeeze(0).cpu().numpy(), threshold)
        else:
            heat_map, mask, vis_img = visualize(img, scores[i], threshold)
        """
        frame = pack_visualize(heat_map, mask, vis_img, scores, get_image_crop_size())
        dirname, path = os.path.split(test_list[i])
        output_path = "result/"+path
        cv2.imwrite(output_path, frame)       
        result_list.append(output_path)

    listsResult.set(result_list)
    load_detail(result_list[0], False)
    ListboxResult.select_set(0)

def test_from_video(net, params, train_outputs, threshold):
    result_path = "result.mp4"

    video_path = test_folder
    if video_path == "camera":
        global test_list, output_index
        video_path = test_list[output_index].split(":")[1]

    capture = webcamera_utils.get_capture(video_path)
    f_h = int(get_image_crop_size())
    f_w = int(get_image_crop_size()) * 3
    writer = webcamera_utils.get_writer(result_path, f_h, f_w)

    score_map = []

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = preprocess(img, get_image_resize(), keep_aspect=get_keep_aspect(), crop_size=get_image_crop_size())
        if EnableOptimization:
            dist_tmp = infer_optimized(net, params, train_outputs, img, get_image_crop_size(), device, logger, weights_torch), 
            score_map.append(dist_tmp)
            roi_img = None
            scores = normalize_scores_torch(score_map, get_image_crop_size(), roi_img)
            

        else:
            dist_tmp = infer(net, params, train_outputs, img, get_image_crop_size())

            score_map.append(dist_tmp)
            roi_img = None
            scores = normalize_scores(score_map, get_image_crop_size(), roi_img)    # min max is calculated dynamically, please set fixed min max value from calibration data for production

            heat_map, mask, vis_img = visualize(denormalization(img[0]), scores[len(scores)-1], threshold)
            frame = pack_visualize(heat_map, mask, vis_img, scores, get_image_crop_size())

            cv2.imshow('frame', frame)
        frame_shown = True

        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    global result_list, listsResult, ListboxResult
    result_list = [result_path]
    listsResult.set(result_list)
    load_detail(result_list[0], False)
    ListboxResult.select_set(0)

# ======================
# Select file
# ======================

train_folder = "train"
test_folder = None

def to_file_name(list):
    new_list = []
    for file in list:
        l = file.split("/")
        new_list.append(l[len(l) - 1])
    return new_list

def train_file_dialog():
    global listsInput, ListboxInput, input_index
    global train_folder
    global train_list
    fTyp = [("Image File or Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        train_folder = file_name
        train_list = [file_name]
        listsInput.set(train_list)
        train_index = 0
        ListboxInput.select_set(0)
        load_detail(train_list[0], True)

def train_folder_dialog():
    global listsInput, ListboxInput, input_index
    global train_folder
    global train_list
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askdirectory(initialdir=iDir)
    if len(file_name) != 0:
        train_folder = file_name
        train_list = get_training_file_list()
        listsInput.set(to_file_name(train_list))
        train_index = 0
        ListboxInput.select_set(0)
        if len(train_list)>=1:
            load_detail(train_list[0], True)

def train_camera_dialog():
    global listsInput, ListboxInput, input_index
    global train_folder
    global train_list
    train_folder = "camera"
    train_list = get_camera_list()
    listsInput.set(train_list)
    train_index = 0
    ListboxInput.select_set(0)
    load_detail(train_list[0], True)

def test_file_dialog():
    global listsOutput, ListboxOutput, output_index
    global test_folder
    global test_list
    global test_type
    fTyp = [("Image File or Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        test_folder = file_name
        test_list = [file_name]
        listsOutput.set(test_list)
        test_index = 0
        ListboxOutput.select_set(0)
        if len(test_list)>=1:
            load_detail(test_list[0], False)
        test_type = "video"

def test_folder_dialog():
    global listsOutput, ListboxOutput, output_index
    global test_folder
    global test_list
    global test_type
    global test_roi
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askdirectory(initialdir=iDir)
    if len(file_name) != 0:
        test_folder = file_name
        test_list = get_test_file_list()
        listsOutput.set(to_file_name(test_list))
        test_index = 0
        ListboxOutput.select_set(0)
        load_detail(test_list[0], False)
        test_type = "folder"

def test_camera_dialog():
    global listsOutput, ListboxOutput, output_index
    global test_folder
    global test_list
    global test_type
    test_folder = "camera"
    test_list = get_camera_list()
    listsOutput.set(test_list)
    test_index = 0
    ListboxOutput.select_set(0)
    load_detail(test_list[0], False)
    test_type = "videp"

def test_roi_dialog():
    global test_roi
    fTyp = [("Image File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        test_roi = file_name
        load_roi_image(test_roi)

def save_dialog():
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askdirectory(initialdir=iDir)
    if len(file_name) != 0:
        ret = shutil.copytree('result', file_name, dirs_exist_ok = True)
        print("Copy success to " + file_name)

def get_camera_list():
    index = 0
    inputs = []
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            inputs.append("camera:"+str(index))
        else:
            break
        index=index+1
        cap.release()
    return inputs

def get_file_list(folder):
    base_path = folder+"/"
    files = glob.glob(base_path+"*.jpg")
    files.extend(glob.glob(base_path+"*.png"))
    files.extend(glob.glob(base_path+"*.bmp"))
    image_list = []
    for image_path in files:
        image_list.append(image_path)
    image_list.sort()
    return image_list

def get_training_file_list():
    global train_folder
    return get_file_list(train_folder)

def get_test_file_list():
    global test_folder
    if test_folder!=None:
        return get_file_list(test_folder)
    return ["bottle_000.png"]

def get_result_file_list():
    return []

def get_model_list():
    return ["resnet18 (224)", "resnet18 (448)", "wide_resnet50_2 (224)"]

def get_model_id_list():
    return ["resnet18", "resnet18", "wide_resnet50_2"]

def get_model_resolution_list():
    return [224, 448, 224]

def get_optimization_list():
    return [True, False]
# ======================
# GUI
# ======================

canvas_item = None
canvas_roi_item = None

def main():
    global train_list, test_list, result_list, model_list
    global listsResult, ListboxResult
    global canvas, canvas_roi, scale
    global inputFile, listsInput, input_list, ListboxInput
    global outputFile, listsOutput, output_list, ListboxOutput
    global listsModel, ListboxModel
    global valueKeepAspect, valueCenterCrop


    # rootメインウィンドウの設定
    root = tk.Tk()
    root.title("PaDiM GUI")
    root.geometry("1200x600")

    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=20,pady=10)

    # Listboxの選択肢
    train_list = get_training_file_list()
    test_list = get_test_file_list()
    result_list = get_result_file_list()
    model_list = get_model_list()
    

    listsInput = tk.StringVar(value=train_list)
    listsOutput = tk.StringVar(value=test_list)
    listsResult = tk.StringVar(value=result_list)
    listsModel = tk.StringVar(value=model_list)
    
    

    # 各種ウィジェットの作成
    ListboxInput = tk.Listbox(frame, listvariable=listsInput, width=20, height=12, selectmode=tk.BROWSE, exportselection=False)
    ListboxOutput = tk.Listbox(frame, listvariable=listsOutput, width=20, height=12, selectmode=tk.BROWSE, exportselection=False)
    ListboxResult = tk.Listbox(frame, listvariable=listsResult, width=20, height=12, selectmode=tk.BROWSE, exportselection=False)
    ListboxModel = tk.Listbox(frame, listvariable=listsModel, width=20, height=6, selectmode=tk.BROWSE, exportselection=False)
    

    # Bind the listbox selection event to the toggle_optimization function
    
    ListboxInput.bind("<<ListboxSelect>>", input_changed)
    ListboxOutput.bind("<<ListboxSelect>>", output_changed)
    ListboxResult.bind("<<ListboxSelect>>", result_changed)
    ListboxModel.bind("<<ListboxSelect>>", model_changed)

    ListboxInput.bind("<Double-Button-1>", input_double_click)
    ListboxOutput.bind("<Double-Button-1>", output_double_click)
    ListboxResult.bind("<Double-Button-1>", result_double_click)
    

    ListboxInput.select_set(input_index)
    ListboxOutput.select_set(output_index)
    ListboxResult.select_set(result_index)
    ListboxModel.select_set(model_index)
    

    textRun = tk.StringVar(frame)
    textRun.set("Train")

    textStop = tk.StringVar(frame)
    textStop.set("Test")

    textTrainFolder = tk.StringVar(frame)
    textTrainFolder.set("Open folder")

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Open video")

    textTrainCamera = tk.StringVar(frame)
    textTrainCamera.set("Open camera")

    textTestFolder = tk.StringVar(frame)
    textTestFolder.set("Open folder")

    textTestVideo = tk.StringVar(frame)
    textTestVideo.set("Open video")

    textTestCamera = tk.StringVar(frame)
    textTestCamera.set("Open camera")

    textInput = tk.StringVar(frame)
    textInput.set("Train images")

    textOutput = tk.StringVar(frame)
    textOutput.set("Test images")

    textResult = tk.StringVar(frame)
    textResult.set("Result images")

    textModelDetail = tk.StringVar(frame)
    textModelDetail.set("Preview")

    textRoi = tk.StringVar(frame)
    textRoi.set("ROI")

    textTestRoi = tk.StringVar(frame)
    textTestRoi.set("Open roi image")

    textCheckbox = tk.StringVar(frame)
    textCheckbox.set("Train settings")

    textModel = tk.StringVar(frame)
    textModel.set("Feature extractor model")

    textTestSettings = tk.StringVar(frame)
    textTestSettings.set("Test settings")

    textSlider = tk.StringVar(frame)
    textSlider.set("threshold")

    textSave = tk.StringVar(frame)
    textSave.set("Save images")

    textOptimization = tk.StringVar(frame)
    textOptimization.set("Set optimization")

    textSaveFile = tk.StringVar(frame)
    textSaveFile.set("Trained file format")

    textBenchmark = tk.StringVar(frame)
    textBenchmark.set("Benchmark mode")

    textDevice = tk.StringVar(frame)
    textDevice.set("Optimization device")

    valueKeepAspect = tkinter.BooleanVar()
    valueKeepAspect.set(True)
    valueCenterCrop = tkinter.BooleanVar()
    valueCenterCrop.set(True)
    chkKeepAspect = tk.Checkbutton(frame, variable=valueKeepAspect, text='keep aspect')
    chkCenterCrop = tk.Checkbutton(frame, variable=valueCenterCrop, text='center crop')

    # 各種ウィジェットの作成
    labelInput = tk.Label(frame, textvariable=textInput)
    labelOutput = tk.Label(frame, textvariable=textOutput)
    labelResult = tk.Label(frame, textvariable=textResult)
    labelModelDetail = tk.Label(frame, textvariable=textModelDetail)
    labelRoi = tk.Label(frame, textvariable=textRoi)
    labelCheckbox = tk.Label(frame, textvariable=textCheckbox)
    labelModel = tk.Label(frame, textvariable=textModel)
    labelTestSettings = tk.Label(frame, textvariable=textTestSettings)
    labelSlider = tk.Label(frame, textvariable=textSlider)
    labelOPt = tk.Label(frame, textvariable=textOptimization)
    labelSaveFile = tk.Label(frame, textvariable=textSaveFile)
    labelBenchmark = tk.Label(frame, textvariable=textBenchmark)
    labelDevice = tk.Label(frame, textvariable=textDevice)


    buttonTrain = tk.Button(frame, textvariable=textRun, command=train_button_clicked, width=14)
    buttonTest = tk.Button(frame, textvariable=textStop, command=test_button_clicked, width=14)

    buttonTrainFolder = tk.Button(frame, textvariable=textTrainFolder, command=train_folder_dialog, width=14)
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=train_file_dialog, width=14)
    buttonTrainCamera = tk.Button(frame, textvariable=textTrainCamera, command=train_camera_dialog, width=14)

    buttonTestFolder = tk.Button(frame, textvariable=textTestFolder, command=test_folder_dialog, width=14)
    buttonTestVideo = tk.Button(frame, textvariable=textTestVideo, command=test_file_dialog, width=14)
    buttonTestCamera = tk.Button(frame, textvariable=textTestCamera, command=test_camera_dialog, width=14)
    buttonTestRoi = tk.Button(frame, textvariable=textTestRoi, command=test_roi_dialog, width=14)

    buttonSave = tk.Button(frame, textvariable=textSave, command=save_dialog, width=14)

    canvas = tk.Canvas(frame, bg="black", width=CANVAS_W, height=CANVAS_H)
    canvas.place(x=0, y=0)

    canvas_roi = tk.Canvas(frame, bg="black", width=CANVAS_H, height=CANVAS_H)
    canvas_roi.place(x=0, y=0)

    load_detail(test_list[0], False)

    var_scale = tk.DoubleVar()
    var_scale.set(slider_index)
    scale = tk.Scale(
        frame,
        variable=var_scale,
        orient=tk.HORIZONTAL,
        tickinterval=20,
        length=200,
    )
    scale.bind("<ButtonRelease-1>", slider_changed)

    # 各種ウィジェットの設置
    labelInput.grid(row=0, column=0, sticky=tk.NW, rowspan=1)
    ListboxInput.grid(row=1, column=0, sticky=tk.NW, rowspan=4)
    buttonTrainFolder.grid(row=6, column=0, sticky=tk.NW)
    buttonTrainVideo.grid(row=7, column=0, sticky=tk.NW)
    buttonTrainCamera.grid(row=8, column=0, sticky=tk.NW)

    labelOutput.grid(row=0, column=1, sticky=tk.NW)
    ListboxOutput.grid(row=1, column=1, sticky=tk.NW, rowspan=4)
    buttonTestFolder.grid(row=6, column=1, sticky=tk.NW)
    buttonTestVideo.grid(row=7, column=1, sticky=tk.NW)
    buttonTestCamera.grid(row=8, column=1, sticky=tk.NW)
    labelRoi.grid(row=9, column=1, sticky=tk.NW, columnspan=1)
    canvas_roi.grid(row=10, column=1, sticky=tk.NW, rowspan=4, columnspan=1)
    buttonTestRoi.grid(row=15, column=1, sticky=tk.NW)

    labelResult.grid(row=0, column=2, sticky=tk.NW)
    ListboxResult.grid(row=1, column=2, sticky=tk.NW, rowspan=4)
    buttonSave.grid(row=6, column=2, sticky=tk.NW)

    labelModelDetail.grid(row=0, column=3, sticky=tk.NW, columnspan=3)
    canvas.grid(row=1, column=3, sticky=tk.NW, rowspan=4, columnspan=3)

    buttonTrain.grid(row=6, column=3, sticky=tk.NW)
    buttonTest.grid(row=6, column=4, sticky=tk.NW)

    labelCheckbox.grid(row=8, column=3, sticky=tk.NW)
    chkKeepAspect.grid(row=9, column=3,  sticky=tk.NW)
    chkCenterCrop.grid(row=10, column=3,  sticky=tk.NW)

    labelModel.grid(row=11, column=3, sticky=tk.NW)
    ListboxModel.grid(row=12, column=3, sticky=tk.NW, rowspan=4)

    labelTestSettings.grid(row=8, column=4, sticky=tk.NW, columnspan=3)
    labelSlider.grid(row=9, column=4, sticky=tk.NW, columnspan=3)
    scale.grid(row=10, column=4, sticky=tk.NW, columnspan=3)

    options = ["False", "True"]
    listsOPt = tk.StringVar(value=options)
    labelOPt.grid(row=11, column=0, sticky=tk.NW, columnspan=3)
    ListboxOptimization = tk.Listbox(frame, listvariable=listsOPt, width=20, height=len(options), selectmode=tk.BROWSE, exportselection=False)
    ListboxOptimization.grid(row=11, column=0, padx=0, pady=20)

    # Set the initial selection in the Listbox
    initial_selection = options.index("True")  # Default to "True"
    ListboxOptimization.select_set(initial_selection)
    ListboxOptimization.event_generate('<<ListboxSelect>>')  # Trigger the event to set the initial state

    # Bind the listbox selection event to the toggle_optimization function
    ListboxOptimization.bind('<<ListboxSelect>>', enable_optimization)

    fileOptions = ["pkl", "pt", "npy"]
    listsFileOPt = tk.StringVar(value=fileOptions)
    labelSaveFile.grid(row=12, column=0, sticky=tk.NW, columnspan=3)
    ListboxFileSelect = tk.Listbox(frame, listvariable=listsFileOPt, width=20, height=len(fileOptions), selectmode=tk.BROWSE, exportselection=False)
    ListboxFileSelect.grid(row=12, column=0, padx=0, pady=20)

    # Set the initial selection in the Listbox
    initial_selectionFile = fileOptions.index("pkl")  # Default to "pkl"
    ListboxFileSelect.select_set(initial_selectionFile)
    ListboxFileSelect.event_generate('<<ListboxSelect>>')  # Trigger the event to set the initial state

    # Bind the listbox selection event to the save_type_select function
    ListboxFileSelect.bind('<<ListboxSelect>>', save_type_select)


    fileDevice = ["cpu", "cuda:0", "mps"]
    listsFileDevice = tk.StringVar(value=fileDevice)
    labelDevice.grid(row=11, column=4, sticky=tk.NW)
    ListboxDeviceSelect = tk.Listbox(frame, listvariable=listsFileDevice, width=20, height=len(fileDevice), selectmode=tk.BROWSE, exportselection=False)
    ListboxDeviceSelect.grid(row=11, column=4, padx=0, pady=20)

    # Set the initial selection in the Listbox
    initial_selection_Device = fileDevice.index("cpu")  # Default to "False"
    ListboxDeviceSelect.select_set(initial_selection_Device)
    ListboxDeviceSelect.event_generate('<<ListboxSelect>>')  # Trigger the event to set the initial state

    # Bind the listbox selection event to the save_type_select function
    ListboxDeviceSelect.bind('<<ListboxSelect>>', select_device)
    # メインフレームの作成と設置
    

    fileBenchmark = ["True", "False"]
    listsFileBenchmark = tk.StringVar(value=fileBenchmark)
    labelBenchmark.grid(row=11, column=5, sticky=tk.NW)
    ListboxFileBenchmark = tk.Listbox(frame, listvariable=listsFileBenchmark, width=20, height=len(fileOptions), selectmode=tk.BROWSE, exportselection=False)
    ListboxFileBenchmark.grid(row=11, column=5, padx=0, pady=20)

    # Set the initial selection in the Listbox
    initial_selection_benchmark = fileBenchmark.index("False")  # Default to "False"
    ListboxFileBenchmark.select_set(initial_selection_benchmark)
    ListboxFileBenchmark.event_generate('<<ListboxSelect>>')  # Trigger the event to set the initial state

    # Bind the listbox selection event to the save_type_select function
    ListboxFileBenchmark.bind('<<ListboxSelect>>', enable_benchmark)
    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=20, pady=10)

    root.mainloop()

    

if __name__ == '__main__':
    main()



