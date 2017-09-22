% Function used in conjunction with slider to update Gamma posterior plot
function update_plot_gamma(i,N,x_vec,s,sigma_sq_ML_norm,a_o,b_o)
    % Update parameters
    % As before, take the average of the ML estimate for lambda
    sigma_sq_ML_norm_avg = mean(sigma_sq_ML_norm(:,i),1);
    a_N = a_o(s) + N(i)/2;
    b_N = b_o(s) + N(i)/2*sigma_sq_ML_norm_avg;
    % Update distribution
    p_post = pdf('gamma',x_vec,a_N,b_N);
    % "Plot" graph
    plot(x_vec,p_post)
    title(sprintf('Gamma Posterior Density (a_{0}=%.3f, b_{0}=%.3f)',a_o(s),b_o(s)))
    xlabel('X')
    axis([0 5 0 1])
    legend(['N = ' num2str(N(i))])
    %drawnow
    % Capture plot frame
    M(i) = getframe(gcf);
end