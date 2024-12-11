data {

    int<lower=2> n_parties;
    int<lower=1> n_strata;
    int<lower=1> n_factors;
    int<lower=1> n_coefs;
    real<lower=n_parties - 1> prior_df;

    int<lower=1,upper=n_coefs> coef_idx[n_strata, n_factors];
    int<lower=1,upper=n_coefs> lo[n_factors];
    int<lower=1,upper=n_coefs> hi[n_factors];
    int<lower=0> counts[n_strata, n_parties];
}


parameters {

    cov_matrix[n_parties - 1] cov_factor[n_factors];
    matrix[n_parties - 1, n_coefs] coefs_raw;
    vector[n_parties - 1] intercept_raw;
}


transformed parameters {

    cholesky_factor_cov[n_parties - 1] cf_cov_factor[n_factors];
    vector[n_parties] intercept = append_row(intercept_raw, rep_vector(0, 1));
    matrix[n_parties, n_coefs] coefs;

    for (j in 1:n_factors) {
        cf_cov_factor[j] = cholesky_decompose(cov_factor[j]);
        coefs[,lo[j]:hi[j]] = append_row(cf_cov_factor[j] * coefs_raw[,lo[j]:hi[j]], rep_row_vector(0, hi[j] - lo[j] + 1));
    }
}


model {

    for (j in 1:n_factors)
        cov_factor[j] ~ inv_wishart(prior_df, diag_matrix(rep_vector(prior_df, n_parties - 1)));

    to_vector(coefs_raw) ~ std_normal();
    to_vector(intercept_raw) ~ std_normal();

    for (i in 1:n_strata)
        counts[i] ~ multinomial(softmax(intercept + coefs[,coef_idx[i]] * rep_vector(1, n_factors)));
}
