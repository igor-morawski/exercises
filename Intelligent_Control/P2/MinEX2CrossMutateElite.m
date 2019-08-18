popu_size=70;
bit_length=40;
gene_no=3;
range=[-40 -50 -60;
        40 50 60];
fitfcn='GA_fitfunP2min';
generation_no=200;
crossover_rates = [0.3; 0.7];
mutate_rates = [0.02; 0.08];
elites = [0; 1];


global MIN_offset

t=1:generation_no;
figure_no = 1;

for cr_index=1:length(crossover_rates)
    crossover_rate = crossover_rates(cr_index);
    mutate_rate = mutate_rates(2);
    elite = elites(2);
    [popu, popu_real, fitness, upper, average, lower, BEST_popu]...
        =GA_genetic(popu_size, bit_length, gene_no, range, fitfcn, ...
        generation_no, crossover_rate, mutate_rate, elite);

    disp(['CR = ', num2str(crossover_rate)]);
    minfitness=MIN_offset-upper;
    [minimum_f,generation]=min(minfitness)
    minimum_x=BEST_popu(generation,1)
    minimum_y=BEST_popu(generation,2)
    minimum_z=BEST_popu(generation,3)
    
    figure(figure_no);
    figure_no = figure_no + 1;
    plot(t,minfitness,'*:')
    title({['Minimum of f(x) for CR = ', num2str(crossover_rate)],['Min f = ',num2str(minimum_f)],['Generation no = ',num2str(generation)]})
    xlabel('Generation')
    ylabel('f(x)')
    saveas(gcf,strcat(num2str(figure_no-1),'.jpg'));
end

for mr_index=1:length(mutate_rates)
    mutate_rate = mutate_rates(mr_index);
    crossover_rate = crossover_rates(1);
    elite = elites(2);
    [popu, popu_real, fitness, upper, average, lower, BEST_popu]...
        =GA_genetic(popu_size, bit_length, gene_no, range, fitfcn, ...
        generation_no, crossover_rate, mutate_rate, elite);

    disp(['MR = ', num2str(mutate_rate)]);
    minfitness=MIN_offset-upper;
    [minimum_f,generation]=min(minfitness)
    minimum_x=BEST_popu(generation)
    
    figure(figure_no);
    figure_no = figure_no + 1;
    plot(t,minfitness,'*:')
    title({['Minimum of f(x) for MR = ', num2str(mutate_rate)],['Min f = ',num2str(minimum_f)],['Generation no = ',num2str(generation)]});
    xlabel('Generation')
    ylabel('f(x)')
    saveas(gcf,strcat(num2str(figure_no-1),'.jpg'));
end

for e_index=1:length(elites)
    mutate_rate = mutate_rates(2);
    crossover_rate = crossover_rates(1);
    elite = elites(e_index);
    [popu, popu_real, fitness, upper, average, lower, BEST_popu]...
        =GA_genetic(popu_size, bit_length, gene_no, range, fitfcn, ...
        generation_no, crossover_rate, mutate_rate, elite);

    disp(['Elite = ', num2str(elite)]);
    minfitness=MIN_offset-upper;
    [minimum_f,generation]=min(minfitness)
    minimum_x=BEST_popu(generation,1)
    minimum_y=BEST_popu(generation,2)
    minimum_z=BEST_popu(generation,3)
    
    figure(figure_no);
    figure_no = figure_no + 1;
    plot(t,minfitness,'*:')
    title({['Minimum of f(x) for Elite = ', num2str(elite)],['Min f = ',num2str(minimum_f)],['Generation no = ',num2str(generation)]});
    xlabel('Generation')
    ylabel('f(x)')
    saveas(gcf,strcat(num2str(figure_no-1),'.jpg'));
end