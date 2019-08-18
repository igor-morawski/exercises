function [scalefitness, temptfitness] = GA_linear_fit_scale(fitness,cc)

if cc>=1.5 && cc<=3
    fave=mean(fitness);
    fmax=max(fitness);
    if abs(fmax-fave)>0.1
        a=(cc-1)*fave/(fmax-fave);
        b=(1-a)*fave;
        scalefitness=a*fitness+b;
        temptfitness=scalefitness;
    else
        scalefitness=fitness;
        temptfitness=scalefitness;
    end;
else
    scalefitness=fitness;
    temptfitness=scalefitness;
end;

I=find(scalefitness<=0);
if ~isempty(I)
    Minf=min(scalefitness);
    scalefitness=scalefitness-Minf;
end;