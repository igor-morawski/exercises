x=-40:0.01:40;
y=-50:0.01:50;
z=-60:0.01:60;
q = (x.^2+y.^2+z.^2+x.*2-10)/(cos(y.^2)-sin(z.^2)+x.*y);
plot(q,x,y,z)