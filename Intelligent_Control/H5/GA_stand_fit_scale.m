function scalefitness = GA_stand_fit_scale(fitness,cc)

if cc>=1.5 && cc<=5
    fave=mean(fitness);
    sigma=std(fitness);
    
    scalefitness=fitness-(fave-cc*sigma);
    
else
    scalefitness=fitness;
end;