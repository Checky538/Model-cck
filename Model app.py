import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import joblib
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
model = joblib.load('CO2_adsorption_model.pkl')
# 创建 Tkinter 窗口
root = tk.Tk()
root.title('Prediction of CO2 adsorption')
root.geometry('1600x900')
# 配置列和行
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_columnconfigure(3, weight=1)
root.grid_columnconfigure(4, weight=1)
root.grid_columnconfigure(5, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_rowconfigure(5, weight=1)
root.grid_rowconfigure(6, weight=1)
root.grid_rowconfigure(7, weight=1)
root.grid_rowconfigure(8, weight=1)
root.grid_rowconfigure(9, weight=1)
root.grid_rowconfigure(10, weight=1)
# 生物质组成输入标签
biomass = tk.Label(root, text='Biomass properties', font=('Times New Roman', 16), fg='red')
biomass.grid(row=0, column=0)
biomass_Ash= tk.Label(root, text='Ash content (%)', font=('Times New Roman', 15))
biomass_Ash.grid(row=1, column=0)
biomass_C = tk.Label(root, text='C content (%)', font=('Times New Roman', 15))
biomass_C.grid(row=2, column=0)
biomass_H = tk.Label(root, text='H content (%)', font=('Times New Roman', 15))
biomass_H.grid(row=3, column=0)
biomass_N = tk.Label(root, text='N content (%)', font=('Times New Roman', 15))
biomass_N.grid(row=4, column=0)
biomass_O = tk.Label(root, text='O content (%)', font=('Times New Roman', 15))
biomass_O.grid(row=5, column=0)
# 生物炭组成输入槽位
Ash = tk.StringVar()
Ash_entry = tk.Entry(root, textvariable=Ash, font=('Times New Roman', 15))
Ash_entry.grid(row=1, column=1)
C = tk.StringVar()
C_entry = tk.Entry(root, textvariable=C, font=('Times New Roman', 15))
C_entry.grid(row=2, column=1)
H = tk.StringVar()
H_entry = tk.Entry(root, textvariable=H, font=('Times New Roman', 15))
H_entry.grid(row=3, column=1)
N = tk.StringVar()
N_entry = tk.Entry(root, textvariable=N, font=('Times New Roman', 15))
N_entry.grid(row=4, column=1)
O = tk.StringVar()
O_entry = tk.Entry(root, textvariable=O, font=('Times New Roman', 15))
O_entry.grid(row=5, column=1)
#热解条件输入标签 
Pyrolysis=tk.Label(root,text='Pyrolysis condition',font=('Times New Roman',16),fg='blue')
Pyrolysis.grid(row=0,column=2)
HR=tk.Label(root,text='Heating ratio',font=('Times New Roman',15))
HR.grid(row=1,column=2)
PT=tk.Label(root,text='Pyrolysis temperature (℃)',font=('Times New Roman',15))
PT.grid(row=2,column=2)
Pt=tk.Label(root,text='Pyrolysis time (min)',font=('Times New Roman',15))
Pt.grid(row=3,column=2)
#热解条件输入槽位
HR=tk.StringVar()
HR_entry=tk.Entry(root,textvariable=HR,font=('Times New Roman',15))
HR_entry.grid(row=1,column=3)
PT=tk.StringVar()
PT_entry=tk.Entry(root,textvariable=PT,font=('Times New Roman',15))
PT_entry.grid(row=2,column=3)
Pt=tk.StringVar()
Pt_entry=tk.Entry(root,textvariable=Pt,font=('Times New Roman',15))
Pt_entry.grid(row=3,column=3)
#活化条件输入标签
Activation=tk.Label(root,text='Activation condition',font=('Times New Roman',16),fg='yellow')
Activation.grid(row=0,column=4)
Seq=tk.Label(root,text='Activation step',font=('Times New Roman',15))
Seq.grid(row=1,column=4)
Agent=tk.Label(root,text='Activator',font=('Times New Roman',15))
Agent.grid(row=2,column=4)
It=tk.Label(root,text='Impregnation time (min)',font=('Times New Roman',15))
It.grid(row=3,column=4)
IT=tk.Label(root,text='Impregnation temperature (℃)',font=('Times New Roman',15))
IT.grid(row=4,column=4)
Ratio=tk.Label(root,text='Activator ratio',font=('Times New Roman',15))
Ratio.grid(row=5,column=4)
Flow=tk.Label(root,text='Activatior flow (ml)',font=('Times New Roman',15))
Flow.grid(row=6,column=4)
AT=tk.Label(root,text='Activation temperature (℃)',font=('Times New Roman',15))
AT.grid(row=7,column=4)
At=tk.Label(root,text='Activation time (min)',font=('Times New Roman',15))
At.grid(row=8,column=4)
#活化条件输入槽位
Seq=tk.StringVar()
Seq_entry=tk.Entry(root,textvariable=Seq,font=('Times New Roman',15))
Seq_entry.grid(row=1,column=5)
Agent=tk.StringVar()
Agent_entry=tk.Entry(root,textvariable=Agent,font=('Times New Roman',15))
Agent_entry.grid(row=2,column=5)
It=tk.StringVar()
It_entry=tk.Entry(root,textvariable=It,font=('Times New Roman',15))
It_entry.grid(row=3,column=5)
IT=tk.StringVar()
IT_entry=tk.Entry(root,textvariable=IT,font=('Times New Roman',15))
IT_entry.grid(row=4,column=5)
Ratio=tk.StringVar()
Ratio_entry=tk.Entry(root,textvariable=Ratio,font=('Times New Roman',15))
Ratio_entry.grid(row=5,column=5)
Flow=tk.StringVar()
Flow_entry=tk.Entry(root,textvariable=Flow,font=('Times New Roman',15))
Flow_entry.grid(row=6,column=5)
APT=tk.StringVar()
APT_entry=tk.Entry(root,textvariable=APT,font=('Times New Roman',15))
APT_entry.grid(row=7,column=5)
APt=tk.StringVar()
APt_entry=tk.Entry(root,textvariable=APt,font=('Times New Roman',15))
APt_entry.grid(row=8,column=5)
# 吸附条件输入标签
Adsorption = tk.Label(root, text='Adsorption condition', font=('Times New Roman', 16), fg='green')
Adsorption.grid(row=6, column=0)
T = tk.Label(root, text='Adsorption temperature (℃)', font=('Times New Roman', 15))
T.grid(row=7, column=0)
P = tk.Label(root, text='Adsorption pressure (bar)', font=('Times New Roman', 15))
P.grid(row=8, column=0)
# 吸附条件输入槽位
T = tk.StringVar()
T_entry = tk.Entry(root, textvariable=T, font=('Times New Roman', 15))
T_entry.grid(row=7, column=1)
P = tk.StringVar()
P_entry = tk.Entry(root, textvariable=P, font=('Times New Roman', 15))
P_entry.grid(row=8, column=1)
# 吸附能力 输出标签
output = tk.Label(root, text='Outputs', font=('Times New Roman', 18), fg='red')
output.grid(row=6, column=2)
CO2_label = tk.Label(root, text='Adsorption capacity (mmol/g)', font=('Times New Roman', 15))
CO2_label.grid(row=7, column=2)
# 输出槽位
CO2 = tk.StringVar()
CO2_entry = tk.Entry(root, textvariable=CO2, font=('Times New Roman', 15))
CO2_entry.grid(row=7, column=3)
# 判断输入是否为数字
def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
# 预测按钮函数
def start():
    try:
        inputs = [
            float(Ash_entry.get()), float(C_entry.get()), float(H_entry.get()), float(N_entry.get()), float(O_entry.get()),
            float(HR_entry.get()), float(PT_entry.get()), float(Pt_entry.get()),
            float(Seq_entry.get()), float(Agent_entry.get()), float(It_entry.get()), float(IT_entry.get()),
            float(Ratio_entry.get()), float(Flow_entry.get()), float(APT_entry.get()), float(APt_entry.get()),
            float(T_entry.get()), float(P_entry.get())
        ]
        X = np.array(inputs).reshape(1, -1)
        prediction = model.predict(X)
        CO2.set(f"{prediction[0]:.4f}")  # 保留四位小数
        print(prediction)

    except ValueError:
        tkinter.messagebox.showerror("Error", "Please input all numbers correctly!")
   
# 预测按钮
button = tk.Button(root, text='Prediction', command=start, font=('Times New Roman', 20, 'bold'))
button.grid(row=4, column=3)
root.mainloop()