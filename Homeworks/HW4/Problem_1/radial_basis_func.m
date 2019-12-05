%% Radial Basis Function %%
function r_out = radial_basis_func(r_in)
    if r_in == 0
        r_out = 0;
    else
        r_out = r_in^2 * log(r_in);
    end
end
