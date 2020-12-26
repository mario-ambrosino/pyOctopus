clear
listing = dir("../../export/Scores");
shape_listing = size(listing);
N_scores = shape_listing(1);
results_table = table();
% TODO define a table to collect results.
for index = 3:N_scores
    full_path = fullfile(listing(index).folder,listing(index).name);
    table_score = readtable(full_path);
    % Parameters to Optimize
    THR = table_score.Error_X(1);
    results_table(index-2,1) = table(table_score.Error_X(1));
    DETR = table_score.Detection_Range(1);
    results_table(index-2,2)= table(table_score.Detection_Range(1));
    results_table(index-2,3)= table(table_score.Epsilon(1));
    results_table(index-2,4)= table(table_score.Min_Samples_Cluster(1));
    % Score functions
    PD = mean(table_score.PD);
    results_table(index-2,5)= table(PD);
    ED = mean(table_score.ED);
    results_table(index-2,6) = table(ED);
    PFA = mean(table_score.PFA);
    results_table(index-2,7)= table(PFA);
    % Global Score
    results_table(index-2,8) = table((10 * PD)^2 + (3 * ED)^2 + (2 * (1-PFA))^2 - 10*(THR^2 + DETR^2));
end
results_table.Properties.VariableNames = ["ERX","DER","EPS","MSC","PD","ED","PFA","SCO"];


