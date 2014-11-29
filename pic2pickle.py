# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 19:07:21 2014

@author: Acer
"""
#若改变训练标签请删除原pickle文件或者修复本源代码的BUG！！！！！
import pickle
import cv2
import os
def getbuffer(indexfile,bufname,bufsize,label):
    if (os.path.isfile(bufname)):
        with open(bufname, 'rb') as pf:
            buf=pickle.load(pf)
            pf.close()
        prefix=indexfile.replace(indexfile.split('\\')[-1],'',1)
        f=open(indexfile)
        changed=0
        for i in xrange(bufsize):
            print '\b...loading %f'%((i+1)/(bufsize*1.0))
            line=f.readline()
            if not line:
                break    
            line=line.replace('\n','',1)
            print line
            if i>=len(buf[2]):
                img=cv2.cvtColor(cv2.imread(prefix+line),7)
                assert img.any()
                buf[0].append(img)
                buf[1].append(label)
                buf[2].append(line)
                changed=changed+1
            elif not line==buf[2][i]:
                img=cv2.cvtColor(cv2.imread(prefix+line),7)
                assert img.any()
                buf[0][i]=img
                buf[1][i]=label
                buf[2][i]=line
                changed=changed+1
        with open(bufname, 'wb') as pf:
            pickle.dump(buf,pf)
        print 'changed %d pictrues'%changed
        return buf
    else :
        todump=[[],[],[]]#img,label,filename
        prefix=indexfile.replace(indexfile.split('\\')[-1],'',1)
        f=open(indexfile)
        for i in xrange(bufsize):
            print '\b...loading %f'%((i+1)/(bufsize*1.0))
            line=f.readline()
            if not line:
                break    
            line=line.replace('\n','',1)
            img=cv2.cvtColor(cv2.imread(prefix+line),7)
            assert img.any()
            todump[0].append(img)
            todump[1].append(label)
            todump[2].append(line)
        with open(bufname,'wb') as pf:
            pickle.dump(todump,pf)
        print 'pickle %s dumped'%bufname
    return [todump[0],todump[1]]
def test():
    indexfilename='C:\\Users\\Acer\\Documents\\Visual Studio 2010\\Projects\\face\\mit\\nonfaces\\bg.txt'
    getbuffer(indexfilename,'face_pos_test.pickle',2900,1)
if __name__ == '__main__':
    test()