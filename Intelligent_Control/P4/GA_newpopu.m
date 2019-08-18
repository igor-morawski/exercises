function new_popu = GA_newpopu(popu, fitness, bit_length, crossover_rate, mutate_rate, elite)

new_popu = popu;
popu_n = size(popu, 1);
gene_number = size(popu, 2)/bit_length;

if elite == 1
    tmp_fitness = fitness;
    [max1, index1] = max(tmp_fitness);
    tmp_fitness(index1) = min(tmp_fitness);
    [max2, index2] = max(tmp_fitness);
end

fitness_rate = fitness/sum(fitness);
fitness_rate_cum = cumsum(fitness_rate);

for i = 1: popu_n/2
    tmp = find(fitness_rate_cum > rand);
    parent1 = popu(tmp(1), :);
    tmp = find (fitness_rate_cum > rand);
    parent2 = popu(tmp(1), :);
    
    if rand < crossover_rate
        crossover_point = ceil(rand*(bit_length - 1));
        K = -bit_length;
        for J = 1 : gene_number
            K = K + bit_length;
            new_popu(i*2 - 1, K+1:K+bit_length) = [parent1(K+1:K+crossover_point) parent2(K+crossover_point+1:K+bit_length)];
            new_popu(i*2,     K+1:K+bit_length) = [parent2(K+1:K+crossover_point) parent1(K+crossover_point+1:K+bit_length)];
        end
    end
end

mask = rand(popu_n, gene_number*bit_length) < 2*rand*mutate_rate;
new_popu = xor(new_popu, mask);

if elite==1
    new_popu([1:2], :) = popu([index1 index2], :);
end