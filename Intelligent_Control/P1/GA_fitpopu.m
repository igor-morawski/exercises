function [fitness, popu_real, popu] = GA_fitpopu(popu, bit_length, range, fitfcn)

popu_size = size(popu, 1);

for I = 1 : popu_size,
    [fitness(I), popu_real(I,:)]=GA_fiteach(popu(I,:),bit_length, range, fitfcn);
end

T=-fitness;
[T, index]=sort(T);
fitness=-T;

popu_real = popu_real(index,:);
popu = popu(index,:);