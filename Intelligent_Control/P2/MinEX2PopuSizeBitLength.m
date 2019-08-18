
gene_no=3;
range=[-40 -50 -60;
        40 50 60];
fitfcn='GA_fitfunP2min';
generation_no=200;
crossover_rates = [0.7; 0.9];
mutate_rates = [0.02; 0.08];
elites = [0; 1];
popu_sizes = [30; 100];
bit_lengths = [30; 60];


global MIN_offset

t=1:generation_no;
figure_no = 1;

for ps_index=1:length(popu_sizes)
    popu_size = popu_sizes(ps_index);
    bit_length = bit_lengths(2);
    crossover_rate = crossover_rates(1);
    mutate_rate = mutate_rates(2);
    elite = elites(2);
    [popu, popu_real, fitness, upper, average, lower, BEST_popu]...
        =GA_genetic(popu_size, bit_length, gene_no, range, fitfcn, ...
        generation_no, crossover_rate, mutate_rate, elite);

    disp(['popu_size = ', num2str(popu_size)]);
    minfitness=MIN_offset-upper;
    [minimum_f,generation]=min(minfitness)
    minimum_x=BEST_popu(generation,1)
    minimum_y=BEST_popu(generation,2)
    minimum_z=BEST_popu(generation,3)
    
    figure(figure_no);
    figure_no = figure_no + 1;
    plot(t,minfitness,'*:');
    title({['Minimum of f(x) for popu size = ', num2str(popu_size)],['Min f = ',num2str(minimum_f)],['Generation no = ',num2str(generation)]});
    xlabel('Generation')
    ylabel('f(x)')
    saveas(gcf,strcat(num2str(figure_no-1),'.jpg'));
end

for bls_index=1:length(bit_lengths)
    popu_size = popu_sizes(2);
    bit_length = bit_lengths(bls_index);
    crossover_rate = crossover_rates(1);
    mutate_rate = mutate_rates(1);
    elite = elites(2);
    [popu, popu_real, fitness, upper, average, lower, BEST_popu]...
        =GA_genetic(popu_size, bit_length, gene_no, range, fitfcn, ...
        generation_no, crossover_rate, mutate_rate, elite);

    disp(['bit_length = ', num2str(bit_length)]);
    minfitness=MIN_offset-upper;
    [minimum_f,generation]=min(minfitness)
    minimum_x=BEST_popu(generation,1)
    minimum_y=BEST_popu(generation,2)
    minimum_z=BEST_popu(generation,3)
    
    figure(figure_no);
    figure_no = figure_no + 1;
    plot(t,minfitness,'*:');
    title({['Minimum of f(x) for bit length = ', num2str(bit_length)],['Min f = ',num2str(minimum_f)],['Generation no = ',num2str(generation)]});
    xlabel('Generation')
    ylabel('f(x)')
    saveas(gcf,strcat(num2str(figure_no-1),'.jpg'));
end