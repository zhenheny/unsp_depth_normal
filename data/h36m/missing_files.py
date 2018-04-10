import numpy as np
import scipy.misc as sm
import os
import glob

subjects = ['S1','S5','S6','S7','S8']

for sub in subjects:
   videos = os.listdir(sub+'/frames/')
   for video in videos:
      if video[-1] == "1":
         if len(os.listdir(sub+'/frames/'+video)) != len(os.listdir(sub+'/frames/'+video.split('.')[0]+'.60457274')):
            frames = glob.glob(sub+'/frames/'+video+'/*.jpg')
            txts1 = glob.glob(sub+'/frames/'+video.split('.')[0]+'.60457274/*_cam.txt')
            txts2 = glob.glob(sub+'/frames/'+video.split('.')[0]+'.60457274/*_cam_ext.txt')
            for frame in frames:
               frame_id = frame.split('/')[-1].split('.')[0]
               if not os.path.exists(sub+'/frames/'+video.split('.')[0]+'.60457274/'+frame_id+'.jpg'):
                  img1 = sm.imresize(sm.imread('../'+sub+'/frames/'+video.split('.')[0]+'.60457274/'+'%04d' % (int(frame_id)-3)+'.jpg'), [832,832])
                  img2 = sm.imresize(sm.imread('../'+sub+'/frames/'+video.split('.')[0]+'.60457274/'+frame_id+'.jpg'), [832,832])
                  if os.path.exists('../'+sub+'/frames/'+video.split('.')[0]+'.60457274/'+'%04d' % (int(frame_id)+3)+'.jpg'):
                     img3 = sm.imresize(sm.imread('../'+sub+'/frames/'+video.split('.')[0]+'.60457274/'+'%04d' % (int(frame_id)+3)+'.jpg'), [832,832])
                  else:
                     print('../'+sub+'/frames/'+video.split('.')[0]+'.60457274/'+'%04d' % (int(frame_id)+3)+'.jpg')
                     continue
                  img = np.concatenate([img1, img2, img3], axis=1)
                  sm.imsave(sub+'/frames/'+video.split('.')[0]+'.60457274/'+frame_id+'.jpg', img)
                  os.system('cp '+txts1[0]+' '+sub+'/frames/'+video.split('.')[0]+'.60457274/'+frame_id+'_cam.txt')
                  os.system('cp '+txts1[0]+' '+sub+'/frames/'+video.split('.')[0]+'.60457274/'+frame_id+'_cam_ext.txt')
                  print(sub+'/frames/'+video.split('.')[0]+'.60457274/'+frame_id+'.jpg')

      if video[-1] == "8":
         if len(os.listdir(sub+'/frames/'+video)) != len(os.listdir(sub+'/frames/'+video.split('.')[0]+'.54138969')):
            frames = glob.glob(sub+'/frames/'+video+'/*.jpg')
            txts1 = glob.glob(sub+'/frames/'+video.split('.')[0]+'.54138969/*_cam.txt')
            txts2 = glob.glob(sub+'/frames/'+video.split('.')[0]+'.54138969/*_cam_ext.txt')
            for frame in frames:
               frame_id = frame.split('/')[-1].split('.')[0]
               if not os.path.exists(sub+'/frames/'+video.split('.')[0]+'.54138969/'+frame_id+'.jpg'):
                  img1 = sm.imresize(sm.imread('../'+sub+'/frames/'+video.split('.')[0]+'.54138969/'+'%04d' % (int(frame_id)-3)+'.jpg'), [832,832])
                  img2 = sm.imresize(sm.imread('../'+sub+'/frames/'+video.split('.')[0]+'.54138969/'+frame_id+'.jpg'), [832,832])
                  if os.path.exists('../'+sub+'/frames/'+video.split('.')[0]+'.54138969/'+'%04d' % (int(frame_id)+3)+'.jpg'):
                     img3 = sm.imresize(sm.imread('../'+sub+'/frames/'+video.split('.')[0]+'.54138969/'+'%04d' % (int(frame_id)+3)+'.jpg'), [832,832])
                  else:
                     print('../'+sub+'/frames/'+video.split('.')[0]+'.54138969/'+'%04d' % (int(frame_id)+3)+'.jpg')
                     continue
                  img = np.concatenate([img1, img2, img3], axis=1)
                  sm.imsave(sub+'/frames/'+video.split('.')[0]+'.54138969/'+frame_id+'.jpg', img)
                  os.system('cp '+txts1[0]+' '+sub+'/frames/'+video.split('.')[0]+'.54138969/'+frame_id+'_cam.txt')
                  os.system('cp '+txts1[0]+' '+sub+'/frames/'+video.split('.')[0]+'.54138969/'+frame_id+'_cam_ext.txt')
                  print(sub+'/frames/'+video.split('.')[0]+'.54138969/'+frame_id+'.jpg')