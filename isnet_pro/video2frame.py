import cv2
import os
import operator
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.signal import argrelextrema
from tqdm import tqdm

def ui_frame2video(image_folder,ouput_dir,fps,mode):
    print("\n IS-NET_pro:frame2video generating...")
    if mode =='.mp4':
        return frame2video(image_folder,ouput_dir,fps)
    elif mode == '.avi':
        return frame2video_alpga(image_folder,ouput_dir,fps)


def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    print(len(x), window_len)
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
 
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))
 
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]
 

class Frame:
    """class to hold information about each frame
    
    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)
 
 
def rel_change(a, b):
   x = (b - a) / max(a, b)
   print(x)
   return x


def video2frame(video_path,output_folder,aim_fps_checkbox,aim_fps,time_range_checkbox,start_time,end_time,keyframe_checkbox,len_window):
    print("\n IS-NET_pro:video2frame generating...")

    # 读取视频文件
    # video_path = 'path/to/video.mp4'
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error opening video file")

    # 创建输出文件夹
    # output_folder = 'path/to/output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    ## 两种情况运行和不允许
    if aim_fps_checkbox:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 这个问题是关于视频转图片的方法。首先，我们需要知道视频的帧率（fps），
        # 即每秒钟播放的帧数。然后，我们可以计算每个输出图片之间的时间间隔，即 
        # 1/fps。接着，我们需要确定每个输出图片所在的时间点。为了做到这一点，我
        # 们可以将输出图片的序号乘以时间间隔，然后将结果乘以视频的帧率，再向下取整
        # ，就可以得到输出图片所对应的视频帧。最后，我们只需要将这些视频帧保存为图片即可。
        total_output_frames = int( total_frames * aim_fps / video_fps)

    # 生成需要输出的帧的索引
        if time_range_checkbox:
            frame_indexes = np.linspace(max(start_frame,0), total_frames - 1, min(int( (end_time-start_time) * aim_fps),end_frame), dtype=np.int)
        else :
            frame_indexes = np.linspace(0, total_frames - 1, total_output_frames, dtype=np.int)
        frame_count = 1
        for i in tqdm(frame_indexes):
        # 设置读取帧的位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            # 读取帧并保存为图片
            ret, frame = cap.read()
            if ret:
                # 指定输出文件名
                output_file = os.path.join(output_folder, f'{frame_count:04d}.png')
                # print('\r geneframe:',output_file,end='')

                # 保存帧到输出文件
                cv2.imwrite(output_file, frame)
                frame_count += 1
    elif(keyframe_checkbox):
        # len_window = int(5)
        curr_frame = None
        prev_frame = None 
        frame_diffs = []
        frames = []
        success, frame = cap.read()
        i = 0 
        while(success):
                luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
                curr_frame = luv
                if curr_frame is not None and prev_frame is not None:
                        #logic here
                        diff = cv2.absdiff(curr_frame, prev_frame)
                        diff_sum = np.sum(diff)
                        diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
                        frame_diffs.append(diff_sum_mean)
                        frame = Frame(i, diff_sum_mean)
                        frames.append(frame)
                prev_frame = curr_frame
                i = i + 1
                success, frame = cap.read()   
        cap.release()
    
        # compute keyframe
        keyframe_id_set = set()
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)
        # save all keyframes as image
        cap = cv2.VideoCapture(str(video_path))
        curr_frame = None
        keyframes = []
        success, frame = cap.read()
        idx = 0
        while(success):
                if idx in keyframe_id_set:
                        name = "keyframe_" + str(idx).zfill(4) + ".png"
                        cv2.imwrite(output_folder + name, frame)
                        keyframe_id_set.remove(idx)
                idx = idx + 1
                success, frame = cap.read()
        cap.release()
    else:
        # 逐帧读取视频并保存到输出文件夹
        frame_count = 1
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(num_frames)):
            # 读取一帧
            ret, frame = cap.read()

            # 检查是否成功读取帧
            if not ret:
                break
            if (i >= start_frame and i <= end_frame) or (not time_range_checkbox):
            # 指定输出文件名
                output_file = os.path.join(output_folder, f'{frame_count:04d}.png')
                # print('\r geneframe:',output_file,end='')

                # 保存帧到输出文件
                cv2.imwrite(output_file, frame)

                # 更新帧计数器
                frame_count += 1

    # 释放视频对象
    cap.release()
    print('\n:) done!')

    return ":) done"
    
def frame2video(image_folder,ouput_dir,fps):
    # 读取图像文件列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    image_files.sort()

    # 获取图像的宽度和高度
    img = cv2.imread(os.path.join(image_folder, image_files[0]),cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape

    # 创建输出视频对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ouput_dir+'/output.mp4', fourcc, fps, (width, height), isColor=True)
    num_images = len(image_files)
    frame_num = 0
    # 逐帧写入视频帧
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)
        frame_num +=1
        # print('\r generating video:',f'{100*frame_num/num_images:5.2f}%',end='')

    # 释放视频对象
    out.release()
    print('\n:) done!')
    return ":) done"


def frame2video_alpga(image_folder,ouput_dir,fps):
    # 读取图像文件列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    image_files.sort()

    # 获取图像的宽度和高度
    img = cv2.imread(os.path.join(image_folder, image_files[0]),cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape

    # 创建输出视频对象
    # 格式表在这里：自己查一下对照表
    # https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(ouput_dir+'/output.avi', fourcc, fps, (width, height), isColor=True)
    num_images = len(image_files)
    frame_num = 0
    # 逐帧写入视频帧
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)
        frame_num +=1
        # print('\r generating video:',f'{100*frame_num/num_images:5.2f}%',end='')

    # 释放视频对象
    out.release()
    print('\n:) done!')
    return ":) done"

if __name__ == '__main__':
    # image_folder = r"D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course2\output4"
    # ouput_dir = r"D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course2\output4"
    # fps = 30
    video2frame(r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course1\luming.mp4',r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course1\output2',True,15,True ,0,1)
