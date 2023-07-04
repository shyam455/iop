from flask import Flask, request, render_template, jsonify
import math
import numpy
import numpy as np
import json

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    xl1=request.form['xl1']
    yl1=request.form['yl1']
    xl2=request.form['xl2']
    yl2=request.form['yl2']
    xl3=request.form['xl3']
    yl3=request.form['yl3']
    xmp=request.form['xmp']
    ymp=request.form['ymp']
    mr=request.form['mr']
    lv=request.form['lv']
    psl=request.form['psl']
    nscb=request.form['nscb']
    dsc=request.form['dsc']
    bs=request.form['bs']
    lcs = numpy.empty(shape=(3, 2), dtype=float)
    for i in range(3):
        lcs[i][0] = float(request.form.get('xl'+f'{i+1}'))
        lcs[i][1] = float(request.form.get('yl'+f'{i+1}'))
    
    mpcs =numpy.empty(shape=2, dtype=float)

    mpcs[0] = float(request.form.get('xmp'))
    mpcs[1] = float(request.form.get('ymp'))
    MR = float(request.form.get('mr'))
    V = float(request.form.get('lv'))
    P_l = float(request.form.get('psl'))
    N = float(request.form.get('nscb'))
    d = float(request.form.get('dsc'))
    r = float(d/2)
    B = float(request.form.get('bs'))
    R = float(B/(2*math.sin(math.pi/N)))
    Req = float(pow((N*r*pow(R,N-1)),1/N))
    H = float(lcs[1][1])
    S = float(lcs[2][0])
    Irms = P_l/(1.73205*V)
    Eo = numpy.empty(shape=3, dtype=float)

    C = float((1+5*((H/S)**2)+4*((H/S)**4))**(1/4))
    F = float((1+((2*H/S)**2))**(1/2))
    Eom = float(((((R*0.01)+(N-1)*(r*0.01))*V)/(N*(r*0.01)*(R*0.01)*1.73205*math.log(2*H/((Req*0.01)*C))))*0.01)
    Eo[0]=Eom
    Eo[2]=Eom
    Ecm = float(((((R*0.01)+(N-1)*(r*0.01))*V)/(N*(r*0.01)*(R*0.01)*1.73205*math.log(2*H/((Req*0.01)*F))))*0.01)
    Eo[1]=Ecm
    
    D=np.array([0,0,0], dtype=float)

    for i in range(3):
     D[i]= float((((mpcs[0]-lcs[i][0])**2)+((mpcs[1]-lcs[i][1])**2))**(1/2))
    AN = numpy.empty(shape=3, dtype=float)
    if N>=3:
     for i in range(3):
        AN[i]=120*math.log10(Eo[i])+55*math.log10(d)-11.4*math.log10(D[i])+26.4*math.log10(N)-128.4
    elif N<3:
     for i in range(3):
        AN[i]=120*math.log10(Eo[i])+55*math.log10(d)-11.4*math.log10(D[i])-115.4
    Z=0
    for i in range(3):
     Z = Z+10**(AN[i]/10)
    AN_Total = 10*math.log10(Z)

    RI = numpy.empty(shape=3, dtype=float)
    for i in range(3):
     RI[i]=3.5*Eo[i]+6*d-33*math.log10(D[i]/20)-30
    RI.sort()
    RI_T = float(0)
    if(RI[2]>=(RI[1]+3)):
     RI_T=RI[2]
    else:
     RI_T= ((RI[1]+RI[2])/2)+1.5
    
    D_d=np.array([0,0,0], dtype=float)
    for i in range(3):
     D_d[i]= float((((mpcs[0]-lcs[i][0])**2)+((mpcs[1]+lcs[i][1])**2))**(1/2))
    P=numpy.empty(shape=(3, 3), dtype=float)
    for i in range(3):
      for j in range(3):
        if i==j:
            P[i][i]= float(math.log(2*H/(Req*0.01)))
        else:
            P[i][j]= float((1/2)*math.log((((lcs[i][0]-lcs[j][0])**2)+((lcs[i][1]+lcs[j][1])**2))/(((lcs[i][0]-lcs[j][0])**2)+((lcs[i][1]-lcs[j][1])**2))))
    M = np.linalg.inv(P)
    J=numpy.empty(3,dtype=float)
    K=numpy.empty(3,dtype=float)
    K_M=numpy.empty(3,dtype=float)
    J_M=numpy.empty(3,dtype=float)
    for i in range(3):
      K_M[i]= float(((mpcs[1]+lcs[i][1])/(((mpcs[0]-lcs[i][0])**2)+((mpcs[1]+lcs[i][1])**2)))-((mpcs[1]-lcs[i][1])/(((mpcs[0]-lcs[i][0])**2)+((mpcs[1]-lcs[i][1])**2))))
      J_M[i]= float(((mpcs[0]-lcs[i][0])/(((mpcs[0]-lcs[i][0])**2)+((mpcs[1]-lcs[i][1])**2)))-((mpcs[0]-lcs[i][0])/(((mpcs[0]-lcs[i][0])**2)+((mpcs[1]+lcs[i][1])**2))))
    Hh_T= float(((Irms*1000)/(2*math.pi))*(((K_M[0]*K_M[0])+(K_M[1]*K_M[1])+(K_M[2]*K_M[2])-(K_M[0]*K_M[1])-(K_M[1]*K_M[2])-(K_M[2]*K_M[0]))**(1/2)))
    Hv_T= float(((Irms*1000)/(2*math.pi))*(((J_M[0]*J_M[0])+(J_M[1]*J_M[1])+(J_M[2]*J_M[2])-(J_M[0]*J_M[1])-(J_M[1]*J_M[2])-(J_M[2]*J_M[0]))**(1/2)))
    MF_TM=float((((Hh_T)**2)+((Hv_T)**2))**(1/2))
    for i in range(3):
      J[i]= float((mpcs[0]-lcs[i][0])*(((D[i])**(-2))-((D_d[i])**(-2))))
      K[i]= float(((mpcs[1]-lcs[i][1])/((D[i])**2))-((mpcs[1]+lcs[i][1])/((D_d[i])**2)))
    J_h=np.array([0,0,0],dtype=float)
    K_v=np.array([0,0,0],dtype=float)
    for i in range(3):
      for j in range(3):
        J_h[i] = J_h[i]+(J[j]*M[j][i])
        K_v[i] = K_v[i]+(K[j]*M[j][i])
    Jh= float(((J_h[0]*J_h[0])+(J_h[1]*J_h[1])+(J_h[2]*J_h[2])-(J_h[0]*J_h[1])-(J_h[1]*J_h[2])-(J_h[2]*J_h[0]))**(1/2))
    Kv= float(((K_v[0]*K_v[0])+(K_v[1]*K_v[1])+(K_v[2]*K_v[2])-(K_v[0]*K_v[1])-(K_v[1]*K_v[2])-(K_v[2]*K_v[0]))**(1/2))
    EF_h=float(Jh*V/1.7321)
    EF_v=float(Kv*V/1.7321)
    EF_TM= float((((EF_h)**2)+((EF_v)**2))**(1/2))

    R_M = np.empty(((4*int(MR))+1), dtype=float)

    for i in range((4*int(MR))+1):
     R_M[i]= -int(MR)+i/2
    data_AN = np.empty(((4*int(MR))+1), dtype=float)
    data_RI = np.empty(((4*int(MR))+1), dtype=float)
    data_EF = np.empty(((4*int(MR))+1), dtype=float)
    data_MF = np.empty(((4*int(MR))+1), dtype=float)

    for j in range((4*int(MR))+1):
     D_e=numpy.empty(shape=3, dtype=float)
     D_d_e=numpy.empty(shape=3, dtype=float)
     for k in range(3):
        D_e[k]= float((((R_M[j]-lcs[k][0])**2)+((0-lcs[k][1])**2))**(1/2))
        D_d_e[k]= float((((R_M[j]-lcs[k][0])**2)+((0+lcs[k][1])**2))**(1/2))
     P_e=numpy.empty(shape=(3, 3), dtype=float)
     for k in range(3):
        for m in range(3):
            if k==m:
                P_e[k][k]= float(math.log(2*H/(Req*0.01)))
            else:
                P_e[k][m]= float((1/2)*math.log((((lcs[k][0]-lcs[m][0])**2)+((lcs[k][1]+lcs[m][1])**2))/(((lcs[k][0]-lcs[m][0])**2)+((lcs[k][1]-lcs[m][1])**2))))
     M_e = np.linalg.inv(P_e)
     J_e=numpy.empty(3,dtype=float)
     K_e=numpy.empty(3,dtype=float)

     K_M_e=numpy.empty(3,dtype=float)
     J_M_e=numpy.empty(3,dtype=float)
     for k in range(3):
        K_M_e[k]= float(((0+lcs[k][1])/(((R_M[j]-lcs[k][0])**2)+((0+lcs[k][1])**2)))-((0-lcs[k][1])/(((R_M[j]-lcs[k][0])**2)+((0-lcs[k][1])**2))))
        J_M_e[k]= float(((R_M[j]-lcs[k][0])/(((R_M[j]-lcs[k][0])**2)+((0-lcs[k][1])**2)))-((R_M[j]-lcs[k][0])/(((R_M[j]-lcs[k][0])**2)+((0+lcs[k][1])**2))))
     Hh_T_e= float(((Irms*1000)/(2*math.pi))*(((K_M_e[0]*K_M_e[0])+(K_M_e[1]*K_M_e[1])+(K_M_e[2]*K_M_e[2])-(K_M_e[0]*K_M_e[1])-(K_M_e[1]*K_M_e[2])-(K_M_e[2]*K_M_e[0]))**(1/2)))
     Hv_T_e= float(((Irms*1000)/(2*math.pi))*(((J_M_e[0]*J_M_e[0])+(J_M_e[1]*J_M_e[1])+(J_M_e[2]*J_M_e[2])-(J_M_e[0]*J_M_e[1])-(J_M_e[1]*J_M_e[2])-(J_M_e[2]*J_M_e[0]))**(1/2)))
     data_MF[j]=float(((Hh_T_e)**2+(Hv_T_e)**2)**(1/2))
     for k in range(3):
      J_e[k]= float((R_M[j]-lcs[k][0])*(((D_e[k])**(-2))-((D_d_e[k])**(-2))))
      K_e[k]= float(((0-lcs[k][1])/((D_e[k])**2))-((0+lcs[k][1])/((D_d_e[k])**2)))
     J_h_e=np.array([0,0,0],dtype=float)
     K_v_e=np.array([0,0,0],dtype=float)
     for k in range(3):
        for m in range(3):
            J_h_e[k] = J_h_e[k]+(J_e[m]*M_e[m][k])
            K_v_e[k] = K_v_e[k]+(K_e[m]*M_e[m][k])
     Jh_e= float(((J_h_e[0]*J_h_e[0])+(J_h_e[1]*J_h_e[1])+(J_h_e[2]*J_h_e[2])-(J_h_e[0]*J_h_e[1])-(J_h_e[1]*J_h_e[2])-(J_h_e[2]*J_h_e[0]))**(1/2))
     Kv_e= float(((K_v_e[0]*K_v_e[0])+(K_v_e[1]*K_v_e[1])+(K_v_e[2]*K_v_e[2])-(K_v_e[0]*K_v_e[1])-(K_v_e[1]*K_v_e[2])-(K_v_e[2]*K_v_e[0]))**(1/2))
     data_EF[j]=float(((((Jh_e*V/1.7321)**2)+((Kv_e*V/1.7321)**2))**(1/2)))

     AN_e = numpy.empty(shape=3, dtype=float)
     RI_e = numpy.empty(shape=3, dtype=float)
     
     if N>=3:
        for k in range(3):
            AN_e[k]=120*math.log10(Eo[k])+55*math.log10(d)-11.4*math.log10(D_e[k])+26.4*math.log10(N)-128.4
     elif N<3:
        for k in range(3):
            AN_e[k]=120*math.log10(Eo[k])+55*math.log10(d)-11.4*math.log10(D_e[k])-115.4
    
     Z_e=0
     for k in range(3):
        Z_e = Z_e+10**(AN_e[k]/10)

     AN_Total_e = 10*math.log10(Z_e)
     data_AN[j]=AN_Total_e
     
     for k in range(3):
        RI_e[k]=3.5*Eo[k]+6*d-33*math.log10(D_e[k]/20)-30
    
     RI_e.sort()
     RI_T_e=float(0)
     if(RI_e[2]>=RI_e[1]+3):
       RI_T_e=RI_e[2]
     else:
       RI_T_e=((RI_e[1]+RI_e[2])/2)+1.5
     data_RI[j]=RI_T_e

    labels = R_M.tolist()
    AN_values = data_AN.tolist()
    RI_values=data_RI.tolist()
    EF_values=data_EF.tolist()
    MF_values=data_MF.tolist()
    return render_template("index.html", xl1=xl1, yl1=yl1, xl2=xl2, yl2=yl2, xl3=xl3, yl3=yl3, 
                           xmp=xmp, ymp=ymp, mr=mr, lv=lv, psl=psl, nscb=nscb, dsc=dsc, bs=bs, 
                           lcs=lcs, Eom=Eom, Ecm=Ecm, AN_1=AN[0], AN_2=AN[1], AN_3=AN[2], AN_Total=AN_Total, 
                           RI_1=RI[0], RI_2=RI[1], RI_3=RI[2], RI_Total=RI_T, EF_h=EF_h, EF_v=EF_v, EF_TM=EF_TM, 
                           Hh_T=Hh_T, Hv_T=Hv_T, MF_TM=MF_TM,
                           labels=labels, AN_values=AN_values, RI_values=RI_values, EF_values=EF_values, MF_values=MF_values)


app.run(debug=True)
