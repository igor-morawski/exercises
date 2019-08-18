function PI = GA_fitfunP2min(chro)

global MIN_offset
MIN_offset = 10000;

x = chro(1);
y = chro(2);
z = chro(3);
q = (x^2+y^2+z^2+2*x-10)/(cos(y^2)-sin(z^2)+x*y);

PI = MIN_offset - q;