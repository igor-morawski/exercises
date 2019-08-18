%parameters
R=2.0;
Km=0.1;
J=0.02;
L=0.5;
Kb=0.1;
Kf = 0.01;
K=Kb;
b=Kf;

%gain for fuzzy controler
gain = 10;

%state-space model
A=[-R/L -Kb/L; Km/J -Kf/J];
B=[1/L; 0];
C=[0 1];
D=[0];

%poles for pole-placement
p1 = -3.1315 + 0.8256i;
p2 = -3.1315 - 0.8256i;

%calculate gain for pole-placement
K_PP = place(A,B,[p1 p2]);

%ADD UNCERTAINITY TO PARAMETERS
R=R+(rand()-0.5)*R;
Km=Km+(rand()-0.5)*Km;
J=J+(rand()-0.5)*J;
L=L+(rand()-0.5)*L;
Kb=Kb+(rand()-0.5)*Kb;
Kf = Kf+(rand()-0.5)*Kf;
Kb = Kb+(rand()-0.5)*Kb;
K=Kb;
Kf=Kf+(rand()-0.5)*Kf;
b=Kf;

%gain for fuzzy controler
gain = 10;

%state-space model
A=[-R/L -Kb/L; Km/J -Kf/J];
B=[1/L; 0];
C=[0 1];
D=[0];


%read fuzzy logics model for controller
fzy_mdl = readfis('fzy_mdl');

pidf=readfis('fzy_mdl');
figure
plotmf(pidf,'input',1);
figure
plotmf(pidf,'output',1);
figure
gensurf(pidf)
