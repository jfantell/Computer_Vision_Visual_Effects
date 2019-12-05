function vector_out = clip_points(vector_in,M,N)
    % ensure vectors do not go out of image bounds
    % x
    x_ = vector_in(:,1);
    y_ = vector_in(:,2);
    x_(x_<1) = 1;
    x_(x_>N) = N;
    % y
    y_(y_<1) = 1;
    y_(y_>M) = M;
    vector_out = [x_, y_];
end

