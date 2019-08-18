R=2.0;
Km=0.1;
J=0.02;
L=0.5;
Kb=0.1;
Kf = 0.01;

K=Kb;
b=Kf;

A=[-R/L -Kb/L; Km/J -Kf/J];
B=[1/L; 0];
C=[0 1];
D=[0];