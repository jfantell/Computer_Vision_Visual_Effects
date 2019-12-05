%% GET COLOR VALUES AT SPECIFIED POINTS %%
function [r,g,b] = get_colors(coords,image)
    r = zeros(size(coords,1),1);
    g = zeros(size(coords,1),1);
    b = zeros(size(coords,1),1);
    for i = 1:size(coords,1)
        r(i) = image(coords(i,2),coords(i,1),1);
        g(i) = image(coords(i,2),coords(i,1),2);
        b(i) = image(coords(i,2),coords(i,1),3);
    end
end