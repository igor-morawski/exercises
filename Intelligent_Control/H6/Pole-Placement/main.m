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

poles = eig(A);

p1 = -3.1315 + 0.8256i;
p2 = -3.1315 - 0.8256i;

%K = lqr(A,B,[1 0; 0 1],1);


K = place(A,B,[p1 p2]);

sys_cl = ss(A-B*K,B,C,0);
sys = ss(A,B,C,0);

t = 0:0.01:20;
u = ones(size(t));
%u = sin(2*pi()*0.1*t);

Nbar = rscale(sys, K);
y=lsim(sys_cl,Nbar*u,t);
y=y';
lsimplot(sys_cl,Nbar*u,t)
xlabel('Time (sec)')
ylabel('Angular velocity')
