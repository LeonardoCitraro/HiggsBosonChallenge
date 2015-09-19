function nnupdatefiguresAMS(fhandle, AMS,i, opts)
%NNUPDATEFIGURES updates figures during training
    if i > 1 %dont plot first point, its only a point   
        x_ax = 1:i;
        % create legend
        M            = {'Training','Validation'};

        %create data for plots
        plot_x       = x_ax;
        plot_ye      = AMS;


        %    plotting
        figure(fhandle);   
        p = plot(plot_x,plot_ye, 'linewidth', 2);
        xlabel('Number of epochs'); ylabel('AMS');title('AMS training and validation');
        legend(p, M,'Location','SouthEast');
        set(gca, 'Xlim',[0,opts.numepochs + 1])
        set(gca,'FontSize',14)
        grid
        axis([0 opts.numepochs 2 4]);
        drawnow;
    end
end
