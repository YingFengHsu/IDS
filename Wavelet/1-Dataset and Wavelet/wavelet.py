import pywt
import numpy as np
import json
import csv
import multiprocessing
import signal
import time

p=[]

def signal_handler(signal, frame):
    print("You pressed Ctrl+C!")
    for singlep in p:
        singlep.terminate()

def applyswt(data,waveletname,divcount,filename):
    feature0=[]
    feature1=[]
    feature2=[]
    feature3=[]
    feature4=[]
    feature5=[]
    feature6=[]
    feature7=[]
    feature8=[]
    feature9=[]
    feature10=[]
    feature11=[]
    feature12=[]
    feature13=[]
    feature14=[]
    feature15=[]
    feature16=[]
    feature17=[]
    feature18=[]
    feature19=[]
    feature20=[]
    feature21=[]
    feature22=[]
    feature23=[]
    feature24=[]
    feature25=[]
    feature26=[]
    feature27=[]
    feature28=[]
    feature29=[]
    feature30=[]
    feature31=[]
    feature32=[]
    feature33=[]
    feature34=[]
    feature35=[]
    feature36=[]
    feature37=[]
    feature38=[]
    feature39=[]
    feature40=[]
    feature41=[]
    feature42=[]
    feature43=[]
    feature44=[]
    feature45=[]
    feature46=[]
    feature47=[]
    feature48=[]
    feature49=[]
    feature50=[]
    feature51=[]
    feature52=[]
    feature53=[]
    feature54=[]
    feature55=[]
    feature56=[]
    feature57=[]
    feature58=[]
    feature59=[]
    feature60=[]
    feature61=[]
    feature62=[]
    feature63=[]
    feature64=[]
    feature65=[]
    feature66=[]
    feature67=[]
    feature68=[]
    feature69=[]
    feature70=[]
    feature71=[]
    feature72=[]
    feature73=[]
    feature74=[]
    feature75=[]
    feature76=[]
    for row in data:
        feature0.append(row[0])
        feature1.append(float(row[1]))
        feature2.append(float(row[2]))
        feature3.append(float(row[3]))
        feature4.append(float(row[4]))
        feature5.append(float(row[5]))
        feature6.append(float(row[6]))
        feature7.append(float(row[7]))
        feature8.append(float(row[8]))
        feature9.append(float(row[9]))
        feature10.append(float(row[10]))
        feature11.append(float(row[11]))
        feature12.append(float(row[12]))
        feature13.append(float(row[13]))
        feature14.append(float(row[14]))
        feature15.append(float(row[15]))
        feature16.append(float(row[16]))
        feature17.append(float(row[17]))
        feature18.append(float(row[18]))
        feature19.append(float(row[19]))
        feature20.append(float(row[20]))
        feature21.append(float(row[21]))
        feature22.append(float(row[22]))
        feature23.append(float(row[23]))
        feature24.append(float(row[24]))
        feature25.append(float(row[25]))
        feature26.append(float(row[26]))
        feature27.append(float(row[27]))
        feature28.append(float(row[28]))
        feature29.append(float(row[29]))
        feature30.append(float(row[30]))
        feature31.append(float(row[31]))
        feature32.append(float(row[32]))
        feature33.append(float(row[33]))
        feature34.append(float(row[34]))
        feature35.append(float(row[35]))
        feature36.append(float(row[36]))
        feature37.append(float(row[37]))
        feature38.append(float(row[38]))
        feature39.append(float(row[39]))
        feature40.append(float(row[40]))
        feature41.append(float(row[41]))
        feature42.append(float(row[42]))
        feature43.append(float(row[43]))
        feature44.append(float(row[44]))
        feature45.append(float(row[45]))
        feature46.append(float(row[46]))
        feature47.append(float(row[47]))
        feature48.append(float(row[48]))
        feature49.append(float(row[49]))
        feature50.append(float(row[50]))
        feature51.append(float(row[51]))
        feature52.append(float(row[52]))
        feature53.append(float(row[53]))
        feature54.append(float(row[54]))
        feature55.append(float(row[55]))
        feature56.append(float(row[56]))
        feature57.append(float(row[57]))
        feature58.append(float(row[58]))
        feature59.append(float(row[59]))
        feature60.append(float(row[60]))
        feature61.append(float(row[61]))
        feature62.append(float(row[62]))
        feature63.append(float(row[63]))
        feature64.append(float(row[64]))
        feature65.append(float(row[65]))
        feature66.append(float(row[66]))
        feature67.append(float(row[67]))
        feature68.append(float(row[68]))
        feature69.append(float(row[69]))
        feature70.append(float(row[70]))
        feature71.append(float(row[71]))
        feature72.append(float(row[72]))
        feature73.append(float(row[73]))
        feature74.append(float(row[74]))
        feature75.append(float(row[75]))
        feature76.append(float(row[76]))
    
    swtfeature1=pywt.swt(feature1,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature2=pywt.swt(feature2,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature3=pywt.swt(feature3,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature4=pywt.swt(feature4,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature5=pywt.swt(feature5,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature6=pywt.swt(feature6,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature7=pywt.swt(feature7,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature8=pywt.swt(feature8,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature9=pywt.swt(feature9,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature10=pywt.swt(feature10,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature11=pywt.swt(feature11,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature12=pywt.swt(feature12,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature13=pywt.swt(feature13,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature14=pywt.swt(feature14,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature15=pywt.swt(feature15,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature16=pywt.swt(feature16,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature17=pywt.swt(feature17,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature18=pywt.swt(feature18,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature19=pywt.swt(feature19,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature20=pywt.swt(feature20,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature21=pywt.swt(feature21,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature22=pywt.swt(feature22,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature23=pywt.swt(feature23,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature24=pywt.swt(feature24,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature25=pywt.swt(feature25,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature26=pywt.swt(feature26,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature27=pywt.swt(feature27,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature28=pywt.swt(feature28,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature29=pywt.swt(feature29,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature30=pywt.swt(feature30,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature31=pywt.swt(feature31,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature32=pywt.swt(feature32,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature33=pywt.swt(feature33,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature34=pywt.swt(feature34,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature35=pywt.swt(feature35,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature36=pywt.swt(feature36,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature37=pywt.swt(feature37,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature38=pywt.swt(feature38,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature39=pywt.swt(feature39,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature40=pywt.swt(feature40,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature41=pywt.swt(feature41,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature42=pywt.swt(feature42,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature43=pywt.swt(feature43,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature44=pywt.swt(feature44,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature45=pywt.swt(feature45,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature46=pywt.swt(feature46,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature47=pywt.swt(feature47,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature48=pywt.swt(feature48,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature49=pywt.swt(feature49,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature50=pywt.swt(feature50,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature51=pywt.swt(feature51,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature52=pywt.swt(feature52,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature53=pywt.swt(feature53,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature54=pywt.swt(feature54,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature55=pywt.swt(feature55,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature56=pywt.swt(feature56,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature57=pywt.swt(feature57,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature58=pywt.swt(feature58,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature59=pywt.swt(feature59,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature60=pywt.swt(feature60,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature61=pywt.swt(feature61,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature62=pywt.swt(feature62,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature63=pywt.swt(feature63,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature64=pywt.swt(feature64,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature65=pywt.swt(feature65,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature66=pywt.swt(feature66,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature67=pywt.swt(feature67,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature68=pywt.swt(feature68,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature69=pywt.swt(feature69,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature70=pywt.swt(feature70,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature71=pywt.swt(feature71,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature72=pywt.swt(feature72,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature73=pywt.swt(feature73,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature74=pywt.swt(feature74,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature75=pywt.swt(feature75,waveletname,trim_approx=True,level=divcount,norm=False)
    swtfeature76=pywt.swt(feature76,waveletname,trim_approx=True,level=divcount,norm=False)
    
    middlefeatures=[]
    middlefeatures.append(feature0)
    for swf in [swtfeature1,swtfeature2,swtfeature3,swtfeature4,swtfeature5,swtfeature6,swtfeature7,swtfeature8,swtfeature9,swtfeature10,swtfeature11,swtfeature12,swtfeature13,swtfeature14,swtfeature15,swtfeature16,swtfeature17,swtfeature18,swtfeature19,swtfeature20,swtfeature21,swtfeature22,swtfeature23,swtfeature24,swtfeature25,swtfeature26,swtfeature27,swtfeature28,swtfeature29,swtfeature30,swtfeature31,swtfeature32,swtfeature33,swtfeature34,swtfeature35,swtfeature36,swtfeature37,swtfeature38,swtfeature39,swtfeature40,swtfeature41,swtfeature42,swtfeature43,swtfeature44,swtfeature45,swtfeature46,swtfeature47,swtfeature48,swtfeature49,swtfeature50,swtfeature51,swtfeature52,swtfeature53,swtfeature54,swtfeature55,swtfeature56,swtfeature57,swtfeature58,swtfeature59,swtfeature60,swtfeature61,swtfeature62,swtfeature63,swtfeature64,swtfeature65,swtfeature66,swtfeature67,swtfeature68,swtfeature69,swtfeature70,swtfeature71,swtfeature72,swtfeature73,swtfeature74,swtfeature75,swtfeature76]:
        for i in range(divcount+1):
            middlefeatures.append(swf[i])
    
    outputfeatures=np.array(middlefeatures).T.tolist()
    
    with open("FullTest-"+filename+'-'+str(waveletname)+"-"+str(divcount)+".txt", "w") as fp:
        json.dump(outputfeatures, fp)
    
    print("FullTest-"+filename+'-'+str(waveletname)+"-"+str(divcount)+".txt Saved")

def openfile(filename,waveletname):
    inputfile=open('/data/lee/newdataset/'+filename+'.csv')
    reader=csv.reader(inputfile)
    OriginalData=[row for row in reader]
    timerfp=open("FullTest-"+filename+'-'+str(waveletname)+"-time.txt", "w")
    for i in range(5):
        start = time.time()
        applyswt(OriginalData,waveletname,i+1,filename)
        end = time.time()
        print(str(i)+':', file=timerfp)
        print(str(end-start), file=timerfp)
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    for myarg in ['DoS-Test','DoS-TrainVali']:
        for myarg2 in ['haar','coif3','sym5']:
            p.append(multiprocessing.Process(target=openfile, args=(myarg,myarg2,)))
    for singlep in p:
        singlep.start()

    for singlep in p:
        singlep.join()
