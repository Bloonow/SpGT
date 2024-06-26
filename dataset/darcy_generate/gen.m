num_file = 2;
num_data = 1024;
S = 1024;

for id_file = 1: num_file
    filename = "darcy_R" + string(S) + "_N" + string(num_data) + "_" + string(id_file) + ".mat";
    disp("==== " + filename + " ====")
    a = zeros(num_data, S, S);  % thresh_a
    u = zeros(num_data, S, S);  % thresh_p

    for id_data = 1: num_data
        disp("Generating " + string(id_data) + " Datum ...")
        [a_val, u_val] = Darcy_Gen(S);  % thresh_a, thresh_p
        a(id_data,:,:) = a_val;
        u(id_data,:,:) = u_val;
    end
    a_mean = mean(a);
    u_mean = mean(u);
    a_std = std(a);
    u_std = std(u);

    save(filename, 'a', 'u', 'a_mean', 'u_mean', 'a_std', 'u_std', '-v7.3');
    disp("==== SAVE " + filename + " DONE ====")
end
