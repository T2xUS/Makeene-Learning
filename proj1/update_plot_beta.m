% Function used in conjunction with slider to update Beta posterior plot
function update_plot_beta(i,N,p_vec,s,m,l,a,b)
    % Update parameters
    A_post = m(i)+a(s);
    B_post = l(i)+b(s);
    % Update distribution
    p_post = pdf('beta',p_vec,A_post,B_post);
    % "Plot" graph
    plot(p_vec,p_post)
    title(sprintf('Beta Posterior Density (A_{prior}=%.3f, B_{prior}=%.3f)',a(s),b(s)))
    xlabel('Probability of Success')
    ylabel('Density')
    axis([0 1 0 25])
    legend(['N = ' num2str(N(i))])
end