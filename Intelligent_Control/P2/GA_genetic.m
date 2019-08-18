function [popu, popu_real, fitness, upper, average, lower, BEST_popu]...
    =GA_genetic(popu_size, bit_length, gene_no, range, fitfcn, ...
    generation_no, crossover_rate, mutate_rate, elite)

initpopu=GA_initpopu(popu_size, bit_length, gene_no);
popu=initpopu;

upper=      zeros(generation_no,1);
average=    zeros(generation_no,1);
lower=      zeros(generation_no,1);
BEST_popu=  zeros(generation_no,gene_no);

for nn = 1:generation_no;
    [fitness, popu_real, popu] = GA_fitpopu(popu, bit_length, range, fitfcn);
    
        [upper(nn),index]=max(fitness);
        average(nn)=mean(fitness);
        lower(nn)=min(fitness);
        BEST_popu(nn,1:gene_no)=popu_real(index,:); 

    popu=GA_newpopu(popu, fitness, bit_length, crossover_rate, mutate_rate, elite);
end
