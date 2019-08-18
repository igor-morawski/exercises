global MIN_offset Kp Ki Kd t y aa bb cc


popu_size=30;
bit_length=40;
gene_no=3;
range=[0 0 0; 30 30 30];
fitfcn='GA_fitfun_P5PID';
generation_no=50;
crossover_rate=0.8;
mutate_rate=0.02;
elite=1;


[popu, popu_real, fitness, upper, average, lower, BEST_popu]...
    =GA_genetic(popu_size, bit_length, gene_no, range, fitfcn, ...
    generation_no, crossover_rate, mutate_rate, elite);

minfitness=MIN_offset-upper;
[minimum_f,generation]=min(minfitness)
minimum_Kp_Ki_Kd=BEST_popu(generation,:);
Kp=minimum_Kp_Ki_Kd(1)
Ki=minimum_Kp_Ki_Kd(2)
Kd=minimum_Kp_Ki_Kd(3)

%%%%%%%%%%%%%%%%%%%%%
figure(1)
ttt=1:generation_no;
plot(ttt,minfitness,'*:')
title('Minimum of PI=sum(abs(100*1-y(I))))')
xlabel('Generation')
ylabel('PI')
%%%%%%%%%%%%%%%%%%%%%
figure(2)
aa=3;
bb=0;
cc=4;
sim('P5GPID');
subplot(2,2,1)
plot(t,y)
title('aa=3 bb=0 cc=4')
xlabel('time')
ylabel('y')

aa=5+2*(rand-0.5)*2;
bb=1+1*(rand-0.5)*2;
cc=6+2*(rand-0.5)*2;
sim('P5GPID');
subplot(2,2,2)
plot(t,y)
title('random aa, bb, cc')
xlabel('time')
ylabel('y')

aa=5;
bb=1;
cc=6;
sim('P5GPID');
subplot(2,2,3)
plot(t,y)
title('aa=5 bb=1 cc=6')
xlabel('time')
ylabel('y')

aa=7;
bb=2;
cc=8;
sim('P5GPID');
subplot(2,2,4)
plot(t,y)
title('aa=7 bb=2 cc=8')
xlabel('time')
ylabel('y')

%%%%%%%%%%%%%%%%%%%%%
figure(3)
aa=3;
bb=0;
cc=4;
sim('P5GPID');
subplot(2,2,1)
plot(t,u)
title('aa=3 bb=0 cc=4')
xlabel('time')
ylabel('u')

aa=5+2*(rand-0.5)*2;
bb=1+1*(rand-0.5)*2;
cc=6+2*(rand-0.5)*2;
sim('P5GPID');
subplot(2,2,2)
plot(t,u)
title('random aa, bb, cc')
xlabel('time')
ylabel('u')

aa=5;
bb=1;
cc=6;
sim('P5GPID');
subplot(2,2,3)
plot(t,u)
title('aa=5 bb=1 cc=6')
xlabel('time')
ylabel('u')

aa=7;
bb=2;
cc=8;
sim('P5GPID');
subplot(2,2,4)
plot(t,u)
title('aa=7 bb=2 cc=8')
xlabel('time')
ylabel('u')

