clear
clc
 
%{
%plot y=f(x)
x=-10:0.01:10;
%y=sin(pi()*x*0.5)+cos(2*pi()*x)+sin(pi()*x.^2)+cos(pi()*x.^2);
y=-2*pi()*x.*sin(pi()*x.^2)+2*pi()*x.*cos(pi()*x.^2)-2*pi()*sin(2*pi()*x)+(pi()/2)*cos((pi()*x)/2);
plot(x,y,'k','LineWidth',1.5)
xlabel('x')
ylabel('y')
title('sin(0.5\pix)+cos(2\pix)+sin(\pix^2)+cos(\pix^2)')
grid on
%}

learning_rate_Array = [0.001, 0.01, 0.1, 1];
x0_Array = -10:0.1:10;
iterations = 100;
for i_x0 = 1:numel(x0_Array)
    for i_alpha = 1:numel(learning_rate_Array)
        learning_rate = learning_rate_Array(i_alpha);
        x(1)=x0_Array(i_x0);
        epsilon = 0.0001;
        for i=1:iterations
            dy(i)=-2*pi()*x(i).*sin(pi()*x(i)^2)+2*pi()*x(i)*cos(pi()*x(i)^2)...
                -2*pi()*sin(2*pi()*x(i))+(pi()/2)*cos((pi()*x(i))/2);
            if abs(dy(i)) <= epsilon
                y(i)=sin(pi()*x(i)*0.5)+cos(2*pi()*x(i))+sin(pi()*x(i)^2)+cos(pi()*x(i)^2);
                [n,xx,yy,dyy] = deal(i,x(i),y(i),dy(i));
                disp(['Results for x0 = ', num2str(x(1)), ', alpha = ', num2str(learning_rate) ...
                    ': n = ', num2str(n), ', ', 'xx = ', num2str(xx), ...
                    ', ', 'yy = ', num2str(yy), ', ', 'dyy = ', num2str(dyy)])
                break
            else
                x(i+1)=x(i)-learning_rate*dy(i);
                y(i)=sin(pi()*x(i)*0.5)+cos(2*pi()*x(i))+sin(pi()*x(i)^2)+cos(pi()*x(i)^2);
            end

        if i == iterations
                   % disp(['Results for x0 = ', num2str(x(1)), ', alpha = ', num2str(learning_rate) ...
                   % ': no results found in ', num2str(iterations), ' iterations'])
        end
        end
    end
end