from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
# Create your views here.


#! importing model to APP

import sklearn
import joblib

model = joblib.load('./models/DECISION_TREE.pkl')

def index(request):
    context={'a':"hello world"}
    return render(request,'index.html',context)
    # return HttpResponse({'a':1})


def predictLoan(request):
    if(request.method=='POST'):
        temp={}
        temp['Gender']=request.POST.get('GenderVal')
        temp['Married']=request.POST.get('MarriedVal')
        temp['Education']=request.POST.get('EducationVal')
        temp['Self_Employed']=request.POST.get('Self_EmployedVal')
        temp['ApplicantIncome']=request.POST.get('ApplicantIncomeVal')
        temp['CoapplicantIncome']=request.POST.get('CoapplicantIncomeVal')
        temp['LoanAmount']=request.POST.get('LoanAmountVal')
        temp['Loan_Amount_Term']=request.POST.get('Loan_Amount_TermVal')
        temp['Credit_History']=request.POST.get('Credit_HistoryVal')
        temp['Urban']=request.POST.get('UrbanVal')
        temp['Rural']=request.POST.get('RuralVal')
        temp['Semiurban']=request.POST.get('SemiurbanVal')
        temp['0']=request.POST.get('zeroVal')
        temp['1']=request.POST.get('oneVal')
        temp['2']=request.POST.get('twoVal')
        temp['3']=request.POST.get('threeVal')






    testData= pd.DataFrame({'x':temp}).transpose()
    scoreVal=model.predict(testData)[0]
    if(scoreVal==1):
        context={'scoreVal':'Eligible'}
    else:
        context={'scoreVal':'Not Eligible'}
    return render(request,'index.html',context)