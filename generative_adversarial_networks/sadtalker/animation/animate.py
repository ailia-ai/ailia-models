import os
import cv2
import numpy as np
import imageio
from pydub import AudioSegment
from skimage import img_as_ubyte

from animation.make_animation import make_animation
from animation.face_enhancer import enhancer_generator_with_len, enhancer_list
from animation.paste_pic import paste_pic
from animation.videoio import save_video_with_watermark

class AnimateFromCoeff:
    def __init__(self, generator_net, kp_detector_net, mapping_net, retinaface_net, gfpgan_net, use_onnx):
        self.generator_net = generator_net
        self.kp_detector_net = kp_detector_net
        self.he_estimator_net = None
        self.mapping_net = mapping_net
        self.retinaface_net = retinaface_net
        self.gfpgan_net = gfpgan_net
        self.use_onnx = use_onnx
    
    def generate(self, x, video_save_dir, pic_path, crop_info, 
                 enhancer=None, background_enhancer=None, preprocess='crop', img_size=256):
        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics_list']
        yaw_c_seq = x.get('yaw_c_seq', None)
        pitch_c_seq = x.get('pitch_c_seq', None)
        roll_c_seq = x.get('roll_c_seq', None)
        frame_num = x['frame_num']

        predictions_video = make_animation(
            source_image, source_semantics, target_semantics,
            self.generator_net, self.kp_detector_net, self.he_estimator_net, self.mapping_net, 
            yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp=True,
            use_onnx=self.use_onnx
        )

        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]

        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]
            image = np.transpose(image.data, [1, 2, 0]).astype(np.float32)
            video.append(image)
        result = img_as_ubyte(video)

        ### the generated video is 256x256, so we keep the aspect ratio, 
        original_size = crop_info[0]
        if original_size:
            result = [
                cv2.resize(result_i, (img_size, int(img_size * original_size[1] / original_size[0])))
                for result_i in result
            ]
        
        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)
        imageio.mimsave(path, result, fps=float(25), codec='libx264')

        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path

        audio_path = x['audio_path']
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name + '.wav')

        sound = AudioSegment.from_file(audio_path)
        end_time = frame_num * 1000 / 25
        word = sound.set_frame_rate(16000)[0:end_time]
        word.export(new_audio_path, format="wav")

        save_video_with_watermark(path, new_audio_path, av_path, watermark= False)
        print(f'The generated video is named {video_save_dir}/{video_name}') 

        if 'full' in preprocess.lower():
            # only add watermark to the full image.
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, 
                      extended_crop= True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}') 
        else:
            full_video_path = av_path 

        # paste back then enhancers
        if enhancer:
            video_name_enhancer = x['video_name'] + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_' + video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer) 
            return_path = av_path_enhancer

            try:
                enhanced_images_gen_with_len = enhancer_generator_with_len(
                    full_video_path, method=enhancer, bg_upsampler=background_enhancer,
                    retinaface_net=self.retinaface_net, gfpgan_net=self.gfpgan_net
                )  
            except:
                enhanced_images_gen_with_len = enhancer_list(
                    full_video_path, method=enhancer, bg_upsampler=background_enhancer
                )
            
            imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark= False)
            print(f'The generated video is named {video_save_dir}/{video_name_enhancer}')
            os.remove(enhanced_path)

        os.remove(path)
        os.remove(new_audio_path)

        return return_path
