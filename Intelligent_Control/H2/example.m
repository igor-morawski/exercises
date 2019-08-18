%example:
%population of 1000 for rich variety of solutions
%[fitness, popu_real, popu] = GA_fitpopu(GA_initpopu(1000,8,2), 8, [0 0; 10 10], 'GA_fitfun')



%[fitness, popu_real, popu] = GA_fitpopu([0 0 0 0 0 0 0 0; 0 0 0 0 1 0 1 0; 0 0 1 0 1 0 1 0; ...
%    1 0 1 0 1 0 1 0; 1 1 1 1 1 1 1 1], 4, [0 0; 10 10], 'GA_fitfun')
%Z = GA_funz(popu_real)


[fitness, popu_real, popu] = GA_fitpopu(GA_initpopu(10,8,1),8,[0;10],'GA_fitfunXXX');
for k=1:5
    new_popu = GA_wheel(fitness, popu);
    for i=1:size(new_popu, 1)
        new_popu_real(i) = GA_bit2num(new_popu(i,:), [0;10]);
    end
end


%[fitness, popu_real, popu] = GA_fitpopu(GA_newpopu(popu, fitness, 8, 1, 0.04, 1), 8, [0;1], 'GA_fitfun');