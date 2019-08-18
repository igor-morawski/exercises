%Tracking car simulation
clear;
pidf=readfis('carf2');
subplot(211)
plotmf(pidf,'input',1);
subplot(212)
plotmf(pidf,'input',2);
figure
plotmf(pidf,'output',1);
figure
gensurf(pidf)
%simulation by SIMULINK
sim('car1');
figure
plot(t,y)
xlabel('time(sec)');
ylabel('velocity(km/hr)');
figure
plot(t,y1)
xlabel('time(sec)');
ylabel('target car velocity(km/hr)');
figure
plot(t,a)
xlabel('time(sec)');
ylabel('distance(m)');
figure
plot(t,u)
xlabel('time(sec)');
ylabel('control input');
