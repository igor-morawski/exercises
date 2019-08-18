function [PI, chro_real] = GA_fiteach(chro, bit_length, range, fitfcn)

gene_number = length(chro)/bit_length;

for i=1 : gene_number
    temp = GA_bit2num(chro((i-1)*bit_length+1:i*bit_length), range(:,i));
    chro_real(i) = round(temp*1000000)/1000000;
end

PI = feval(fitfcn, chro_real);

