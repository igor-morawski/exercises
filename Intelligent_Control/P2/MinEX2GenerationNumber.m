gene_no=3;
range=[-40 -50 -60;
        40 50 60];
fitfcn='GA_fitfunP2min';
generation_nos=[100 200 300];
crossover_rates = [0.7; 0.9];
mutate_rates = [0.02; 0.08];
elites = [0; 1];
popu_sizes = [30; 100];
bit_lengths = [30; 60];


global MIN_offset

figure_no = 1;

for gn_index=1:length(generation_nos)
    popu_size = popu_sizes(2);
    bit_length = bit_lengths(2);
    crossover_rate = crossover_rates(1);
    mutate_rate = mutate_rates(2);
    elite = elites(2);
    generation_no=generation_nos(gn_index);
    t=1:generation_no;
    [popu, popu_real, fitness, upper, average, lower, BEST_popu]...
        =GA_genetic(popu_size, bit_length, gene_no, range, fitfcn, ...
        generation_no, crossover_rate, mutate_rate, elite);

    disp(['generation_nos = ', num2str(generation_no)]);
    minfitness=MIN_offset-upper;
    [minimum_f,generation]=min(minfitness)
    minimum_x=BEST_popu(generation,1)
    minimum_y=BEST_popu(generation,2)
    minimum_z=BEST_popu(generation,3)
    
    figure(figure_no);
    figure_no = figure_no + 1;
    plot(t,minfitness,'*:');
    title({['Minimum of f(x) for gen. number = ', num2str(generation_no)],['Min f = ',num2str(minimum_f)],['Generation no = ',num2str(generation)]});
    xlabel('Generation')
    ylabel('f(x)')
    saveas(gcf,strcat(num2str(figure_no-1),'.jpg'));
end