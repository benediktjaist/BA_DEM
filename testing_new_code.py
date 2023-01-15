def calculate_damp_coeff(epsilon_n):
    h_1 =float(0.2446517)
    h_2 =float(-0.5433478)
    h_3 =float(0.9280126)
    h_4 =float(-1.5897793)
    h_5 =float(1.2102729)
    h_6 =float(3.3815393)
    h_7 =float(6.3814014)
    h_8 =float(-34.482428)
    h_9 =float(25.672467)
    h_10 =float(94.396267)

    beta = float(epsilon_n - 0.5)

    #gamma = epsilon_n * (1-epsilon_n)**2 * [h_1 + beta * (h_2 + beta * (h_3 + beta *
                    #(h_4 + beta * (h_5 + beta * (h_6 + beta * (h_7 + beta * (h_8 + beta * (h_9 + beta * h_10))))))))]

    return gamma

x = calculate_damp_coeff(float(0.8))
print(x)

