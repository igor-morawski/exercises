clear
clc

% plot y=f(x)
%x=0:0.01:1;
%y=sin(2*pi()*x)+0.5*sin(6*pi()*x)+0.5*cos(10*pi()*x);
%plot(x,y,'k','LineWidth',1.5)
%xlabel('x')
%ylabel('y')
%title('sin(2\pix)+^{1}/_{2}sin(6\pix)+^{1}/_{2}cos(10\pix)')
%grid on
%clear x y 

%solve
learning_rate_Array = [0.001, 0.01, 0.1, 1];
x0_Array = [0.1, 0.4, 0.9]
iterations = 1000
for i_x0 = 1:numel(x0_Array)
    for i_alpha = 1:numel(learning_rate_Array)
        learning_rate = learning_rate_Array(i_alpha);
        x(1)=x0_Array(i_x0);
        epsilon = 0.0001;
        for i=1:iterations
            dy(i)=-5*pi()*sin(10*pi()*x(i))+3*pi()*cos(6*pi()*x(i))+2*pi()*cos(2*pi()*x(i));

            if abs(dy(i)) <= epsilon
                y(i)=sin(2*pi()*x(i))+0.5*sin(6*pi()*x(i))+0.5*cos(10*pi()*x(i));
                [n,xx,yy,dyy] = deal(i,x(i),y(i),dy(i));
                disp(['Results for x0 = ', num2str(x(1)), ', alpha = ', num2str(learning_rate) ...
                    ': n = ', num2str(n), ', ', 'xx = ', num2str(xx), ...
                    ', ', 'yy = ', num2str(yy), ', ', 'dyy = ', num2str(dyy)])
                break
            else
                x(i+1)=x(i)-learning_rate*dy(i);
                y(i)=sin(2*pi()*x(i))+0.5*sin(6*pi()*x(i))+0.5*cos(10*pi()*x(i));
            end

        if i == iterations
                    disp(['Results for x0 = ', num2str(x(1)), ', alpha = ', num2str(learning_rate) ...
                    ': no results found in ', num2str(iterations), ' iterations'])
        end
        end
    end
end