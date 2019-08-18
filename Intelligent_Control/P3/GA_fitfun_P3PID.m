function PI = GA_fitfun_P3PID(chro)

global MIN_offset Kp Ki Kd t y 

MIN_offset = 10000;

Kp = chro(1);
Ki = chro(2);
Kd = chro(3);

sim('P3GPID');

I=find(t>2);
z=sum(abs(100*(1-y(I))));
if max(y)>1.20
    z=z+1000;
end
if max(u)>20
    z=z+1000;
end
PI=MIN_offset-z;