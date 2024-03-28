import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn import metrics
from django.shortcuts import render
def home(request):
   return render(request,'home.html')
def result(request):
   df=pd.read_csv(r"C:\Users\proni\Downloads\archive (6).zip")
   encoder=LabelEncoder()
   encoder_data=encoder.fit_transform(df[['sex']])
   df['sex']=encoder_data
   df=df. replace({'t':1,'f':0})
   LE=LabelEncoder()
   new_df=LE.fit_transform(df[['Classes']])
   df['Classes']=new_df
   x=df.drop(['Classes'],axis=1)
   y=df['Classes']
   X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)
   smote=SMOTE()
   X_resample,Y_resample=smote.fit_resample(X_train,Y_train)
   model=KNeighborsClassifier(n_neighbors=3)
   model.fit(X_resample,Y_resample)
   val1=float(request.GET['n1'])
   a1=val1
   val2=(request.GET['n2'])
   if val2=='Male':
      a2=1
      a7=0
   else:
      a2=0
      val7=(request.GET['n7'])
      if val7=="True":
         a7=1
      else:
         a7=0
   val3=(request.GET['n3'])
   if val3=="True":
      a3=1
   else:
      a3=0
   val4=(request.GET['n4'])
   if val4=="True":
      a4=1
   else:
      a4=0
   val5=(request.GET['n5'])
   if val5=="True":
      a5=1
   else:
      a5=0
   val6=(request.GET['n6'])
   if val6=="True":
      a6=1
   else:
      a6=0
   val8=(request.GET["n8"])
   if val8=="True":
      a8=1
   else:
      a8=0
   val9=(request.GET["n9"])
   if val9=="True":
      a9=1
   else:
      a9=0
   val10=request.GET['n10']
   if val10=="True":
      a10=1
   else:
      a10=0
   val11=(request.GET['n11'])
   if val11=="True":
      a11=1
   else:
      a11=0
   val12=(request.GET['n12'])
   if val12=="True":
      a12=1
   else:
      a12=0
   val13=(request.GET['n13'])
   if val13=="True":
      a13=1
   else:
      a13=0
   val14=(request.GET['n14'])
   if val14=="True":
      a14=1
   else:
      a14=0
   val15=(request.GET['n15'])
   if val15=="True":
      a15=1
   else:
      a15=0
   val16=(request.GET['n16'])
   if val16=="True":
      a16=1
   else:
      a16=0
   val17=(request.GET['n17'])
   if val17=="True":
      a17=1
      val18=float(request.GET['n18'])
      a18=val18
   else:
      a17=0
      a18=0.00
   val19=(request.GET["n19"])
   if val19=="True":
       a19=1
       val20=float(request.GET["n20"])
       a20=val20
   else:
      a19=0
      a20=0.00
   val21=(request.GET["n21"])
   if val21=="True":
      a21=1
      val22=float(request.GET["n22"])
      a22=val22
   else:
      a21=0
      a22=0.00
   val23=(request.GET["n23"])
   if val23=="True":
      a23=1
      val24=float(request.GET["n24"])
      a24=val24
   else:
      a23=0
      a24=0.00
   val25=(request.GET["n25"])
   if val25=="True":
      a25=1
      val26=float(request.GET["n26"])
      a26=val26
   else:
      a25=0
      a26=0.00
   pred=model.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26]])
   result1=""
   if pred[0]==2:
      result1="You are negative"
   elif pred[0]==1:
       result1="You are Hypothyroidic"
   else:
      result1="You are Hyperthyroidic"
   return render(request,'home.html',{"result2":result1})
   
   
   
   
   

   
   





 