function PI = GA_fitfun(chro)

global MIN_offset

MIN_offset = 10;
x = chro(1);
y = sin(2*pi()*x)+0.5*sin(6*pi()*x)+0.5*cos(10*pi()*x);
PI = MIN_offset - y;