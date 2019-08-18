function real = GA_bit2num(bit, range)

deci = polyval(bit, 2);
real = deci/(2^length(bit)-1)*(range(2)-range(1))+range(1);