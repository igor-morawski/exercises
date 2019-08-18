global MIN_offset Kp Ki Kd t y


popu_size=100;
bit_length=64;
gene_no=3;
range=[0 0 0; 30 30 30];
fitfcn='GA_fitfun_P3PID';
generation_no=60;
crossover_rate=0.8;
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
title('Minimum of PI=sum(abs(100*1-y(I))))')
xlabel('Generation')
ylabel('PI')

figure(2)
sim('P3GPID');
plot(t,y)
title('Step Response')
xlabel('time')
ylabel('y')

figure(3)
plot(t,u)
title('Control Energy')
xlabel('time')
ylabel('u')


minfitness_graph=MIN_offset-upper;
minfitness_best=minimum_f;
y_best=y;
u_best=u;
Kp_best = Kp;
Ki_best = Ki;
Kd_best = Kd;
a=1;
if (max(abs(1-(y(find(t>2))))) > 0.02 || max(u)>20 || max(y)>1.20)
    display(num2str(a))
    display('...')
    P3PID_rp
end