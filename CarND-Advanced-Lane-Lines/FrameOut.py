import cv2
#print(cv2.__version__)

vidcap = cv2.VideoCapture('project_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
    cv2.imwrite("./project/frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    #print 'Read a new frame: ', success

    if count > 2000:
        break


    count += 1