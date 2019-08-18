
gene_no=3;
range=[-40 -50 -60;
        40 50 60];
fitfcn='GA_fitfunP2max';
generation_no=200;
crossover_rates = [0.7; 0.9];
mutate_rates = [0.02; 0.08];
elites = [0; 1];
popu_sizes = [30; 100];
bit_lengths = [30; 60];



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
    maxfitness=upper;
    [maximum_f,generation]=max(maxfitness)
    maximum_x=BEST_popu(generation,1)
    maximum_y=BEST_popu(generation,2)
    maximum_z=BEST_popu(generation,3)
    
    figure(figure_no);
    figure_no = figure_no + 1;
    plot(t,maxfitness,'*:')
    title({['Maximum of f(x) for popu size = ', num2str(popu_size)],['Max f = ',num2str(maximum_f)],['Generation no = ',num2str(generation)]});
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
    maxfitness=upper;
    [maximum_f,generation]=max(maxfitness)
    maximum_x=BEST_popu(generation,1)
    maximum_y=BEST_popu(generation,2)
    maximum_z=BEST_popu(generation,3)
    
    figure(figure_no);
    figure_no = figure_no + 1;
    plot(t,maxfitness,'*:')
    title({['Maximum of f(x) for bit length = ', num2str(bit_length)],['Max f = ',num2str(maximum_f)],['Generation no = ',num2str(generation)]});
    xlabel('Generation')
    ylabel('f(x)')
    saveas(gcf,strcat(num2str(figure_no-1),'.jpg'));
end