import numpy as np  
import pandas as pd  
import os
import shutil


def main():


    SourceDir = "//Volumes/myShare/epson_pxm840f/Scan/"
    TargetDir = "//Volumes/EncryptShared/ins"

    files = os.listdir(SourceDir)    
    pdf_files = [f for f in files if f.split('.')[-1] == "pdf" ]

    print(" file move operation ")
    fcnt = 0
            
    for f in pdf_files:
        #
        # name converted to list 
        fstr = list(f)
        # then pick up first 1 char.
        firstChr = fstr[0].lower()

        print(type(firstChr))
        #
        #  target name dir is ins + first 1 char of file
        #         
        targetnamedir = os.path.join(TargetDir, firstChr)
        Sourcefile = os.path.join(SourceDir,f)

        if os.path.isdir(targetnamedir):
            #print(targetnamedir)
            pass
        else:
            os.makedirs(targetnamedir)

        try:
            print(" %s --> %s  " % (f, targetnamedir))
            shutil.move(Sourcefile, targetnamedir)
            fcnt += 1
        except EnvironmentError:
            print("IO Eror on %s " % f)



    print("total %d files moved." % fcnt)
        
if __name__ == "__main__":
    main()