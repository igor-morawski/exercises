function real = GA_bit2num(bit, range)

real = bi2de(fliplr(bit))/(2^length(bit)-1)*(range(2)-range(1))+range(1);