%% COMPUTE WEIGHTS REQUIRED TO ESTIMATE POINT CORRESPONDENCES %%
function weights = get_weights(known_points_I1,known_points_I2)
    A = create_rbf_matrix(known_points_I1);
    b = cat(1,known_points_I2,zeros(3,2));
    weights = A \ b;
end

%% CONSTRUCT RBF MATRIX (SECTION 5.2.1)
function rbf_matrix = create_rbf_matrix(known_points)
    % create top left submatrix
    N = size(known_points,1);
    TL = zeros(N,N);
    for i = 1:N
        for j = 1:N
            diff = known_points(i,:) - known_points(j,:);
            r_ij = sqrt(diff(1)^2 + diff(2)^2);
            TL(i,j) = radial_basis_func(r_ij); 
        end
    end
    disp(["Norm of Top left submatrix", norm(TL,2)]);
    
    % create top right submatrix
    TR = cat(2,known_points,ones(N,1));
         
    % create bottom left submatrix
    BL = transpose(TR);

    % create bottom right submatrix
    BR = zeros(3);

    % stitch together all sub-matrices
    top = cat(2,TL,TR);
    bottom = cat(2,BL,BR);
    rbf_matrix = cat(1,top,bottom);
    disp(["Norm of A matrix", norm(rbf_matrix,2)]);
end