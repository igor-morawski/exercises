global MIN_offset Kp Ki Kd t y


popu_size=100000;
bit_length=240;
gene_no=3;
range=[0 0 0; 500 500 500];
fitfcn='GA_fitfun_P4PID';
generation_no=5;
crossover_rate=0.5;
mutate_rate=0.08;
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

figure(1)
ttt=1:generation_no;
plot(ttt,minfitness,'*:')
title('Minimum of PI=max(abs(u))')
xlabel('Generation')
ylabel('PI')

figure(2)
sim('P4GPID');
plot(t,y)
title('Step Response')
xlabel('time')
ylabel('y')

figure(3)
plot(t,u)
title('Control Energy')
xlabel('time')
ylabel('u')