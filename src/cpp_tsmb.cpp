//#define ARMA_64BIT_WORD 1
//#define ARMA_USE_CXX11

#include <RcppArmadillo.h>
#include "hf_struct_hmbdims.h"
#include "hf_InitDims.h"
#include "hf_MultInv.h"
#include "hf_MuVar.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;

// Estimators from MacKinnon and H. White (1985)
arma::mat HsMacKinnonInner(
  const arma::mat& MK_resids_1,
  const arma::mat& MK_resids_2,
  const arma::mat& MK_gamma_1,
  const arma::mat& MK_gamma_2,
  const arma::mat& G_mat,
  const double& N_doub
) {
  mat result = (N_doub - 1) / N_doub
    * G_mat * diagmat(MK_resids_1 % MK_resids_2) * G_mat.t()
    - (N_doub - 1) / (N_doub * N_doub)
    * MK_gamma_1 * MK_gamma_2.t();

  return result;
}

arma::mat HsMacKinnon(
  const arma::mat& MK_resids_1,
  const arma::mat& G_mat,
  const int& N_int,
  const bool& useSame = TRUE,
  const arma::mat& MK_resids_2 = arma::zeros(1,1)
) {
  const double N_doub = (double)N_int;

  const mat MK_gamma_1 = G_mat * MK_resids_1;

  if (useSame == TRUE) {
    return HsMacKinnonInner(
      MK_resids_1,
      MK_resids_1,
      MK_gamma_1,
      MK_gamma_1,
      G_mat,
      N_doub
    );
  }

  const mat MK_gamma_2 = G_mat * MK_resids_2;

  return HsMacKinnonInner(
    MK_resids_1,
    MK_resids_2,
    MK_gamma_1,
    MK_gamma_2,
    G_mat,
    N_doub
  );
}


// [[Rcpp::export()]]
Rcpp::List cpp_tsmb(
  const arma::vec& Y_S,
  const arma::mat& X_S,
  const arma::mat& X_Sa,
  const arma::mat& Z_Sa,
  const arma::mat& Z_U) {
  const hmbdims dims = InitDims(Y_S, X_S, X_Sa, Z_Sa, Z_U);

  // Storing for easy access
  const mat Beta_inv_mat = MultInv(X_S.t() * X_S);
  const mat Alpha_inv_mat = MultInv(Z_Sa.t() * Z_Sa);
  const mat G_mat_X = Beta_inv_mat * X_S.t();
  const mat G_mat_Z = Alpha_inv_mat * Z_Sa.t();

  // Calculating Beta and Gamma
  const vec Beta = G_mat_X * Y_S;
  mat Gamma = G_mat_Z * X_Sa;
  Gamma.col(0) = zeros(dims.P_Z + 1); Gamma.at(0, 0) = 1;

  // Calculating residuals
  const vec resids_S = Y_S - X_S * Beta;
  const mat resids_Sa = X_Sa - Z_Sa * Gamma;

  // Declaring G_func_sum and BetaCov
  mat BetaCov, G_func_sum;

  // Initializing phi_const_mat and omega
  mat phi_const_mat = zeros(dims.P_X, dims.P_X);
  double omega = 0;

    omega = accu(resids_S % resids_S) / dims.df_S_X;
    BetaCov = omega * Beta_inv_mat;

    double g_sum = 0;
    for (int k = 1; k <= dims.P_X; ++k) {
      phi_const_mat.at(k - 1, k - 1) =
        as_scalar(resids_Sa.col(k).t() * resids_Sa.col(k))
        / dims.df_Sa_Z;
      g_sum += (Beta.at(k) * Beta.at(k) - BetaCov.at(k, k))
        * phi_const_mat.at(k - 1, k - 1);

      for (int l = 1; l < k; ++l) {
        phi_const_mat.at(k - 1, l - 1) =
          as_scalar(resids_Sa.col(k).t() * resids_Sa.col(l))
          / dims.df_Sa_Z;
        g_sum += 2 * (Beta.at(k) * Beta.at(l) - BetaCov.at(k, l))
          * phi_const_mat.at(k - 1, l - 1);
      }
    }

    G_func_sum = Alpha_inv_mat * g_sum;
  

  mat AlphaCov = Gamma * BetaCov * Gamma.t() + G_func_sum;

  // Getting mu estimation and mu variance estimation
  long double muVar[2]; MuVar(
    muVar,
    Z_U,
    Gamma * Beta,
    AlphaCov
  );

  List ret;
  ret["Beta"] = Beta;
  ret["Gamma"] = Gamma;
  ret["BetaCov"] = BetaCov;
  ret["AlphaCov"] = AlphaCov;
  ret["phis"] = phi_const_mat;
  ret["omega"] = omega;
  ret["mu"] = *muVar;
  ret["muVar"] = *(muVar + 1);
  return ret;
}
