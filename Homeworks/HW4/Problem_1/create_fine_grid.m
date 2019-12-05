%% CREATE FINE GRID %%
function coords = create_fine_grid(image,interval)
    h = ceil(size(image,1)/interval);
    w = ceil(size(image,2)/interval);
    coords = zeros(h*w,2);
    index = 1;
    % iterate through each column (y dimension)
    for x = 1:interval:size(image,2)
        % iterate through each row (x dimension)
        for y = 1:interval:size(image,1)
            coords(index,:) = [x,y];
            index = index + 1;
        end
    end
end