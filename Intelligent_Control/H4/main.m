popu_size=100;
bit_length=32;
gene_no=3;
range=[-40 -50 -60;
        40 50 60];
fitfcn='GA_fitfun';
generation_no=200;
crossover_rate = 0.7;
mutate_rate = 0.19;
elite = 1;

global MIN_offset

[popu, popu_real, fitness, upper, average, lower, BEST_popu]...
    =GA_genetic(popu_size, bit_length, gene_no, range, fitfcn, ...
    generation_no, crossover_rate, mutate_rate, elite);


minfitness=MIN_offset-upper;
[minimum_f,generation]=min(minfitness)
minimum_x=BEST_popu(generation,1)
minimum_y=BEST_popu(generation,2)
minimum_z=BEST_popu(generation,3)

t=1:generation_no;
plot(t,minfitness,'*:')
title('Minimum of f(x)=(x^2+y^2+z^2+2*x-10)/(cos(y^2)-sin(z^2)+x*y))');
xlabel('Generation')
ylabel('f(x)')