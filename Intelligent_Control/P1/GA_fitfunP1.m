function PI = GA_fitfunP1(chro)

global MIN_offset

MIN_offset = 10;
x = chro;
z = sin(pi()*x*0.5)+cos(2*pi()*x)+sin(pi()*x^2)+cos(pi()*x^2);
PI = MIN_offset - z;