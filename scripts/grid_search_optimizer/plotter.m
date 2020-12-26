clf

% Find unique X and Y variables
X = unique(results_table.ERX);
Y = unique(results_table.DER);
C = unique(results_table.EPS);

list = {};
% Generate Matrix
for iC = 1:length(C)
    Z = zeros(length(X),length(Y));
    for iX = 1:length(X)
        for iY = 1:length(Y)
            iZ = find(results_table.ERX == X(iX) & results_table.DER == Y(iY) & results_table.EPS == C(iC) );
            Z(iX,iY) = mean(results_table.SCO(iZ));
        end
    end

    % Interpolate NaNs
    Z = fillmissing(Z,'linear',2,'EndValues','nearest');

    xv = linspace(min(X), max(X), 40);
    yv = linspace(min(Y), max(Y), 40);
    [Yv,Xv] = meshgrid(yv, xv);

    Zv = griddata(Y,X,Z,Yv,Xv,'cubic'); 
    hold on
    s = surf(Xv,Yv,Zv);
    s.CData = iC*ones(size(Zv));
    s.DisplayName = sprintf('EPS = %0.0f',C(iC));
    zmax = max(Zv(:));
    [xmax, ymax] = find(ismember(Zv, max(Zv(:))));
    max_point = [xv(xmax);yv(ymax);zmax];
    list{end+1} = max_point;
    
end

max_holder = cell2mat(list);
s2 = scatter3(max_holder(1,:),max_holder(2,:),max_holder(3,:), "r", 'filled');
s2.DisplayName = "Maximum Points";
hold off
title("Score Landscape")
xlabel("Threshold")
ylabel("Detection Error on Ground")
zlabel("Global Score")
grid off
legend(C, C)