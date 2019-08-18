%example:
%population of 1000 for rich variety of solutions
[fitness, popu_real, popu] = GA_fitpopu(GA_initpopu(1000,8,1), 8, [0;1], 'GA_fitfun');

disp(['Value with the highest fitness: ', num2str(popu_real(1))]);
%[fitness, popu_real, popu] = GA_fitpopu(GA_newpopu(popu, fitness, 8, 1, 0.04, 1), 8, [0;1], 'GA_fitfun');