bit_length = 8;
range = [0; 100];
base_popu = GA_initpopu(10, bit_length, 1);

crossover_rate = [0; 0.7; 0.8; 0.9; 1];
mutate_rate = [0; 0.04; 0.06; 0.08];
elite = [0; 1];
[fitness, popu_real, popu] = GA_fitpopu(base_popu, bit_length, range, 'GA_fitfun3x');
base_fitness = fitness;

disp(['Base population']);
disp(popu);
disp(['Base population real']);
disp(popu_real);
disp(['Base population fitness']);
disp(transpose(fitness));

for cr_index=2:length(crossover_rate)
    [fitness, popu_real, popu] = GA_fitpopu(base_popu, bit_length, range, 'GA_fitfun3x');
    [fitness, popu_real, popu] = GA_fitpopu(GA_newpopu(popu, fitness, bit_length, ...
        crossover_rate(cr_index), mutate_rate(1), elite(1)), bit_length, range, 'GA_fitfun3x');
    disp(['CR = ',num2str(crossover_rate(cr_index))]);
    disp(popu);
    disp(popu_real);
    disp(transpose(fitness));

end

for mr_index=2:length(mutate_rate)
    [fitness, popu_real, popu] = GA_fitpopu(base_popu, bit_length, range, 'GA_fitfun3x');
    [fitness, popu_real, popu] = GA_fitpopu(GA_newpopu(popu, fitness, bit_length, ...
        crossover_rate(1), mutate_rate(mr_index), elite(1)), bit_length, range, 'GA_fitfun3x');
    disp(['CR = ',num2str(mutate_rate(mr_index))]);
    disp(popu);
    disp(popu_real);
    disp(transpose(fitness));
end

 e_index=1;
    [fitness, popu_real, popu] = GA_fitpopu(base_popu, bit_length, range, 'GA_fitfun3x');
    [fitness0, popu_real0, popu0] = GA_fitpopu(GA_newpopu(popu, fitness, bit_length, ...
        crossover_rate(2), mutate_rate(1), elite(e_index)), bit_length, range, 'GA_fitfun3x');
    disp(['E = ',num2str(elite(e_index))]);
    disp(popu0);
    disp(popu_real0);
    disp(transpose(fitness0));
    
 e_index=2;
    [fitness, popu_real, popu] = GA_fitpopu(base_popu, bit_length, range, 'GA_fitfun3x');
    [fitness1, popu_real1, popu1] = GA_fitpopu(GA_newpopu(popu, fitness, bit_length, ...
        crossover_rate(2), mutate_rate(1), elite(e_index)), bit_length, range, 'GA_fitfun3x');
    disp(['E = ',num2str(elite(e_index))]);
    disp(popu1);
    disp(popu_real1);
    disp(transpose(fitness1));

