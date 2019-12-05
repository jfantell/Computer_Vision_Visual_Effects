%% ESTIMATE POINT CORRESPONDENCES %%
function out_points = compute_coorespondences(points_to_estimate,known_points,weights)
    out_points = zeros(size(points_to_estimate,1),2);
    for i = 1:size(points_to_estimate,1)
        out_point_ = compute_correspondence(points_to_estimate(i,1),points_to_estimate(i,2),known_points,weights);
        out_points(i,1) = out_point_(1,1);
        out_points(i,2) = out_point_(1,2);
    end
end

%% COMPUTE CORRESPONDENCES USING F(X,Y) %%
function out_point = compute_correspondence(x,y,known_points,weights)
    sum_r_i_x = 0;
    sum_r_i_y = 0;
    for i = 1:size(known_points,1)
        w1 = weights(i,1);
        w2 = weights(i,2);
        diff = [x,y]-known_points(i,:);
        r_i_ = sqrt(diff(1)^2 + diff(2)^2);
        r_i = radial_basis_func(r_i_);
        sum_r_i_x = sum_r_i_x + (w1 * r_i);
        sum_r_i_y = sum_r_i_y + (w2 * r_i);
    end
    a11 = weights(end-2,1);
    a12 = weights(end-1,1);
    b1 = weights(end,1);
    a21 = weights(end-2,2);
    a22 = weights(end-1,2);
    b2 = weights(end,2);
    out_x = sum_r_i_x + (a11 * x) + (a12 * y) + b1;
    out_y = sum_r_i_y + (a21 * x) + (a22 * y) + b2;
    out_point = [out_x, out_y];
end