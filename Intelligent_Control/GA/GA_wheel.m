function new_popu = GA_wheel(fitness, popu)

Y=min(fitness);
if Y<0
    fitness = fitness - Y;
end
popu_n = length(fitness);

fit_sum = sum(fitness);
fit_mean = fit_sum/popu_n;

popu_select = fitness / fit_mean;

if popu_select(1) < 2
    popu_select = popu_select.^2;
end

Y = sum(popu_select);
while (Y < popu_n)
    I = find(popu_select == 0);
    popu_select(I(1)) = popu_select(I(1))+1;
    Y = sum(popu_select);
end

K=1;

for I=1:popu_n
    if popu_select(I) > 0
        for J = 1 : popu_select(I)
            new_popu(K,:) = popu(I,:);
            K=K+1;
            if K > popu_n
                break
            end
        end
    end
    if K > popu_n
        break
    end
end