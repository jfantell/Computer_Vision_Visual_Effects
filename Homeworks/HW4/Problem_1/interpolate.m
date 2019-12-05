%% Return the interpolated image
function interpolated_image = interpolate(sparse_coordinates_warped, dense_coordinates, color_channel_values,M,N)
    red = color_channel_values(:,1);
    green = color_channel_values(:,2);
    blue = color_channel_values(:,3);
    DT = delaunayTriangulation(sparse_coordinates_warped);
    interpolant_function_red = scatteredInterpolant(DT.Points,red,'natural');
    interpolant_function_green = scatteredInterpolant(DT.Points,green,'natural');
    interpolant_function_blue = scatteredInterpolant(DT.Points,blue,'natural');

    dense_coordinates_interpolated_red = interpolant_function_red(dense_coordinates(:,1),dense_coordinates(:,2));
    dense_coordinates_interpolated_green = interpolant_function_green(dense_coordinates(:,1),dense_coordinates(:,2));
    dense_coordinates_interpolated_blue = interpolant_function_blue(dense_coordinates(:,1),dense_coordinates(:,2));

    dense_coordinates_interpolated = [dense_coordinates_interpolated_red;dense_coordinates_interpolated_green;dense_coordinates_interpolated_blue];
    interpolated_image = reshape(dense_coordinates_interpolated,[M,N,3]);
end