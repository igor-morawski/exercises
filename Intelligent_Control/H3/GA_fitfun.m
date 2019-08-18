function PI = GA_fitfun(chro)

global MIN_offset

MIN_offset = 100;
x = chro(1);
y = chro(2);
z = x*cos(x)+y*cos(y);
PI = MIN_offset - z;