function PI = GA_fitfun_P5PID(chro)

global MIN_offset Kp Ki Kd t y aa bb cc 

MIN_offset = 20000;

Kp = chro(1);
Ki = chro(2);
Kd = chro(3);


aa=3;
bb=0;
cc=4;

sim('P5GPID');
I=find(t>2.5);
z=1.2*sum(abs(100*(1-y(I))));
if max(y)>1.25
    z=z+1000;
end
if max(u)>30
    z=z+1000;
end

aa=5+2*(rand-0.5)*2;
bb=1+1*(rand-0.5)*2;
cc=6+2*(rand-0.5)*2;

sim('P5GPID');
I=find(t>2.5);
z=z+sum(abs(100*(1-y(I))));
if max(y)>1.25
    z=z+1000;
end
if max(u)>30
    z=z+1000;
end

aa=5;
bb=1;
cc=6;

sim('P5GPID');
I=find(t>2.5);
z=z+sum(abs(100*(1-y(I))));
if max(y)>1.25
    z=z+1000;
end
if max(u)>30
    z=z+1000;
end

aa=7;
bb=2;
cc=8;

sim('P5GPID');
I=find(t>2.5);
z=z+sum(abs(100*(1-y(I))));
if max(y)>1.25
    z=z+1000;
end
if max(u)>30
    z=z+1000;
end

PI=MIN_offset-z;