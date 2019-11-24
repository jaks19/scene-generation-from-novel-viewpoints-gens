import os 

imgs = []
for i in range(11498):
    imgs.append(f'/Users/jaks19/Desktop/images/mazes/{i}.png')

def make_video(images, outvid='/Users/jaks19/Desktop/images/mazes/vid.mp4', outimg=None, fps=33, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        
        if not os.path.exists(image):
            continue
            
        if int(image.split('/')[-1].split('.')[0]) > 11000: break

        img = imread(image)
        
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
                
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
            
        vid.write(img)
        
    vid.release()
    return vid

# Use as:
#make_video(imgs, fps=2.1)