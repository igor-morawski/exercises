function new_popu = GA_newpopu(popu, fitness, bit_length, crossover_rate, mutate_rate, elite, range)

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
        
%arithmetic crossover
for i = 1: popu_n/2
    tmp = find(fitness_rate_cum > rand);
    parent1 = popu(tmp(1), :);
    tmp = find (fitness_rate_cum > rand);
    parent2 = popu(tmp(1), :);
    
    if 0 < crossover_rate
        for j=1 : gene_number
            temp = GA_bit2num(parent1((j-1)*bit_length+1:j*bit_length), range(:,j));
            chro_real_parent1(j) = round(temp*1000000)/1000000;
        end
        for j=1 : gene_number
            temp = GA_bit2num(parent2((j-1)*bit_length+1:j*bit_length), range(:,j));
            chro_real_parent2(j) = round(temp*1000000)/1000000;
        end
        %arithmetic crossover
        K = -bit_length;
        alpha = rand;
        for J = 1 : gene_number
            K=K+bit_length;
            %temp=chro_real_parent1(J)+alpha*(chro_real_parent2(J)-chro_real_parent1(J));
            temp=alpha*chro_real_parent1(J)+(1-alpha)*chro_real_parent2(J);
            while (temp>range(2,J) || temp<range(1,J))
                alpha = rand;
                temp=alpha*chro_real_parent1(J)+(1-alpha)*chro_real_parent2(J);
            end
            temp_bi = GA_num2bit(temp, range(:,J), bit_length);
            new_popu(i*2 - 1, K+1:K+bit_length) = temp_bi(1, 1:bit_length);
            temp=alpha*chro_real_parent1(J)+(1-alpha)*chro_real_parent2(J);
            while (temp>range(2,J) || temp<range(1,J))
                alpha = rand;
                temp=alpha*chro_real_parent1(J)+(1-alpha)*chro_real_parent2(J);
            end
            temp_bi = GA_num2bit(temp, range(:,J), bit_length);
            new_popu(i*2, K+1:K+bit_length) = temp_bi(1, 1:bit_length);
        end
    end
end

%mutation
for i = 1: popu_n
    for j=1 : gene_number
        if 20 < mutate_rate
            temp = range(1,j) + rand * (range(2,j)-range(1,j));
            temp_bi = GA_num2bit(temp, range(:,j), bit_length);
            new_popu(i, (gene_number-1)*bit_length+1:gene_number*bit_length) = ...
                temp_bi(1, 1:bit_length);
        end
    end
    
end

if elite==1
    new_popu([1:2], :) = popu([index1 index2], :);
end