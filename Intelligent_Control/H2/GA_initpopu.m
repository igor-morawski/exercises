function initpopu = GA_initpopu(popu_size, bit_length, gene_number)


nn = bit_length*gene_number;
initpopu=round(rand(popu_size, nn));