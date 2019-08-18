function PI = GA_fitfun_P4PID(chro)

global MIN_offset Kp Ki Kd 

MIN_offset = 100000;

Kp = chro(1);
Ki = chro(2);
Kd = chro(3);

sim('P4GPID');

error_ss=-6/(5*Ki);
z = 100*max(abs(u));
if abs(error_ss) > 0.1
    z=z+1000;
end

PI=MIN_offset-z;