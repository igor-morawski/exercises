function Z = GA_funz(popu_real)

for i=1:size(popu_real,1)
    x = popu_real(i,1);
    y = popu_real(i,2);
    Z(i) = x*cos(x)+y*cos(y);
end