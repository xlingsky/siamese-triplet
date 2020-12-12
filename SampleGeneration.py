import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import warnings
import shutil
from sklearn.neighbors import KDTree 
from osgeo import gdal

def readBoundaryFile(filepath):
    boundaries = list()
    fp = open(filepath,'r')
    if fp is None:
        return None
    line = fp.readline()
    boundary_num = int(line)
    for i in range(boundary_num):
        line = fp.readline()
        line = fp.readline()
        points = np.array([float(x) for x in line.split()])
        boundaries.append((points.reshape((points.shape[0]//2,2))))

    return boundaries

def readBuildingFile(filepath):
    boundary = list()
    with open(filepath,'r') as fp:
        for line in fp:
            t = line.split()
            if len(t) < 2:
                break
            boundary.append([float(t[0]),float(t[1])])
    return np.asarray(boundary)

def rasterization(points, xmin, xmax, ymin, ymax, dx, dy):
    cols = int((xmax-xmin)/dx)
    rows = int((ymax-ymin)/dy)

    data = np.full((rows,cols),255,dtype=np.uint8);
    for p in points:
        c = int((p[0]-xmin)//dx)
        r = int((p[1]-ymin)//dy)
        data[r,c] = 0
    return data

def count(xmin, ymin, xmax, ymax, points):
    cnt = 0
    for x,y in points:
        if x >= xmin and x <= xmax and y >= ymin and y<= ymax:
            cnt+=1
    return cnt

def findCorrespondence(query, dataset):
#    xmin, ymin = query.min(axis=0)
#    xmax, ymax = query.max(axis=0)
#    for b in dataset:
#        c = count(xmin, ymin, xmax, ymax, b)
#        p0 = c/(b.shape[0])
#        x0,y0 = b.min(axis=0)
#        x1,y1 = b.max(axis=0)
#        c = count(x0, y0, x1, y1, query)
#        p1 = c/(query.shape[0])
#        if p0>0.4 and p1>0.4:
#            return b
#    return None
    tree = KDTree(query)
    for b in dataset:
        c = tree.two_point_correlation(b,10)
        if c>query.shape[0]*0.8:
            return b
    return None

def angleHist(points, hist):
    if len(points) < 2:
        return hist
    n = len(hist)
    buk = 2*np.pi/n
    centre = np.mean(points,axis=0)
    for x,y in points:
        a = np.arctan2(y-centre[1],x-centre[0])
        if a < 0:
            a = 2*np.pi+a
        i = int(a//buk)
        # print("({},{},{},{})".format(x,y,a,i))
        hist[i] = hist[i]+1
    return hist

def computeMajorDirection(points):
    n = 18
    hist = np.zeros(n)
    clusterSize = 10
    clusterNum = len(points)//clusterSize
    for i in range(clusterNum):
        it = i*clusterSize
        hist = angleHist(points[it:it+clusterSize], hist)
    hist = angleHist(points[clusterNum*clusterSize:], hist)
    t = n//2
    for i in range(t):
        hist[i] = hist[i]+hist[i+t]
    return 2*np.pi/n*(np.argmax(hist)+0.5)

def rotate( points, angle):
    centre = np.mean(points,axis=0)
    cosa = np.cos(angle)
    sina = np.sin(angle)
    r = np.array([[cosa,-sina],[sina,cosa]])
    return np.dot((points-centre),r)+centre
    
def crop(img,percentX,percentY):
    rows,cols = img.shape
    m = np.full(img.shape,255,dtype=np.uint8)
    if percentX == 0:
        percentX = 1
    if percentX>0:
        xmin = 0
        xmax = int(cols*percentX)
    else :
        xmin = int(cols*(1+percentX))
        xmax = cols
    if percentY == 0:
        percentY = 1
    if percentY>0:
        ymin = 0
        ymax = int(rows*percentY)
    else :
        ymin = int(rows*(1+percentY))
        ymax = rows
    m[ymin:ymax,xmin:xmax] = img[ymin:ymax,xmin:xmax]
    return m

def test():
    if os.name == 'posix':
        satellitepath = '/Users/xlingsky/codingspace/OSU/registration/data/satellite_buildings.txt'
        streetpath = '/Users/xlingsky/codingspace/OSU/registration/data/building_lists.txt'
        workdir=''
    else:
        satellitepath = r'J:\xlingsky\temp\crossview\satellite_buildings.txt'
#        streetpath = r'J:\xlingsky\gopro\gopro_result\6_2019\02_10\GX010029\align\2\building_lists.txt'
        streetpath = r'J:\xlingsky\gopro\gopro_result\6_2019\02_10\GX010487\align\2\building_lists.txt'
        workdir=r'J:\xlingsky\temp\crossview'
    idtag = 200
    buildings_t = readBoundaryFile(satellitepath)
    buildings_s = readBoundaryFile(streetpath)
    bnorm_t = list()
    bnorm_s = list()
    idlist = list()
    percentX = [0.6,0.5,-0.6]
    percentY = [0.6,0.5,-0.6]
    for i,b0 in enumerate(buildings_s):
        b1 = findCorrespondence(b0, buildings_t)
        if b1 is None:
            continue
        a = computeMajorDirection(b0)
        b = rotate(b0, a)
        bnorm_s.append(b-np.mean(b,axis=0))
        b = rotate(b1, a)
        bnorm_t.append(b-np.mean(b,axis=0))
        idlist.append(i)

    rect_s = np.array([1e6,1e6,-1e6,-1e6])
    for b in bnorm_s:
        xmin,ymin = np.min(b,axis=0)
        xmax,ymax = np.max(b,axis=0)
        if xmin < rect_s[0]:
            rect_s[0] = xmin
        if ymin < rect_s[1]:
            rect_s[1] = ymin
        if xmax > rect_s[2]:
            rect_s[2] = xmax
        if ymax > rect_s[3]:
            rect_s[3] = ymax
    rect_t = np.array([1e6,1e6,-1e6,-1e6])
    for b in bnorm_t:
        xmin,ymin = np.min(b,axis=0)
        xmax,ymax = np.max(b,axis=0)
        if xmin < rect_t[0]:
            rect_t[0] = xmin
        if ymin < rect_t[1]:
            rect_t[1] = ymin
        if xmax > rect_t[2]:
            rect_t[2] = xmax
        if ymax > rect_t[3]:
            rect_t[3] = ymax
    print('s= {}'.format(rect_s))
    print('t= {}'.format(rect_t))
    rect = np.array([-256,-256,256,256])
#    if rect_s[0] < rect_t[0]:
#        rect[0] = rect_s[0]
#    else:
#        rect[0] = rect_t[0]
#    if rect_s[1] < rect_t[1]:
#        rect[1] = rect_s[1]
#    else:
#        rect[1] = rect_t[1]
#    if rect_s[2] > rect_t[2]:
#        rect[2] = rect_s[2]
#    else:
#        rect[2] = rect_t[2]
#    if rect_s[3] > rect_t[3]:
#        rect[3] = rect_s[3]
#    else:
#        rect[3] = rect_t[3]
#    rect[0]-=5
#    rect[1]-=5
#    rect[2]+=5
#    rect[3]+=5
    for i in range(len(idlist)):
        dir = os.path.join(workdir,'{}'.format(idtag+idlist[i]))
        if not os.path.exists(dir):
            os.mkdir(dir)
        
        img = rasterization(bnorm_s[i],rect[0],rect[2],rect[1],rect[3],1,1)
        path = os.path.join(dir,'streetview_{}.tif'.format(idlist[i]))
        cv2.imwrite(path,img)
        dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
        path = os.path.join(dir,'streetview_dist_{}.tif'.format(idlist[i]))
        cv2.imwrite(path,dist)
        
        img = rasterization(bnorm_t[i],rect[0],rect[2],rect[1],rect[3],1,1)
        path = os.path.join(dir,'topview_{}.tif'.format(idlist[i]))
        cv2.imwrite(path,img)
        dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
        path = os.path.join(dir,'topview_dist_{}.tif'.format(idlist[i]))
        cv2.imwrite(path,dist)
        
        dir = os.path.join(dir,'slicing')
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        for x in percentX:
            for y in percentY:
                m = crop(img,x,y)
                path = os.path.join(dir,'topview_{}_slicing_{}_{}.tif'.format(idlist[i],x,y))
                driver = gdal.GetDriverByName( "GTiff" )
                
                dataset = driver.Create(path,xsize=m.shape[1],ysize=m.shape[0],bands=1,eType=gdal.GDT_Byte)
                dataset.GetRasterBand(1).WriteArray(m)                
                
                dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
                path = os.path.join(dir,'topview_dist_{}_slicing_{}_{}.tif'.format(idlist[i],x,y))
                
                dataset = driver.Create(path,xsize=dist.shape[1],ysize=dist.shape[0],bands=1,eType=gdal.GDT_Float32)
                dataset.GetRasterBand(1).WriteArray(dist)
                dataset = None

    # plt.figure(0)
    # plt.scatter(b0[:,0],b0[:,1])
    # plt.scatter(b1[:,0],b1[:,1])
    # a = computeMajorDirection(b0)
    # t0 = rotate(b0, a)
    # t1 = rotate(b1, a)
    # plt.figure(1)
    # plt.scatter(t0[:,0],t0[:,1])
    # plt.scatter(t1[:,0],t1[:,1])
    # plt.show()
