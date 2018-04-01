import cv2
#print(cv2.__version__)

vidcap = cv2.VideoCapture('test_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
    cv2.imwrite("./test/frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    #print 'Read a new frame: ', success

    #if count > 5000:
    #    break


    count += 1



vidcap = cv2.VideoCapture('project_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
    cv2.imwrite("./project/frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    #print 'Read a new frame: ', success

    #if count > 5000:
    #    break


    count += 1


#for img in clip.iter_frames(progress_bar=True):
#        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#        vimages.append(img)
#        vframes.append("%s - %d" % (file_name, count))
#        count += 1
