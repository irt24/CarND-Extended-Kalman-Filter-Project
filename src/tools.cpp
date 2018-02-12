#include <assert.h>
#include <iostream>
#include <math.h>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

namespace {
const float kSmallFloat = 0.0001;

float enforceNotZero(float x) {
  return fabs(x) < kSmallFloat ? kSmallFloat : x;
}
}  // end anonymous namespace

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  assert(estimations.size() == ground_truth.size());

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  for (int i = 0; i < estimations.size(); i++) {
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array() * residual.array();
      rmse += residual;
  }
  rmse = rmse / estimations.size();  // Calculate the mean.
  return rmse.array().sqrt();
}

void Tools::CalculateJacobian(const VectorXd& x_state, MatrixXd* jacobian) {
  assert(x_state.size() == 4);

  const float px = x_state(0);
  const float py = x_state(1);
  const float vx = x_state(2);
  const float vy = x_state(3);

  const float px2_py2 = px*px + py*py;
  const float px2_py2_1over2 = sqrt(px2_py2);
  const float px2_py2_3over2 = px2_py2 * px2_py2_1over2;

  const float Hj00 = px / enforceNotZero(px2_py2_1over2);
  const float Hj01 = py / enforceNotZero(px2_py2_1over2);
  const float Hj10 = -py / enforceNotZero(px2_py2);
  const float Hj11 = px / enforceNotZero(px2_py2);
  const float Hj20 = py * (vx*py - vy*px) / enforceNotZero(px2_py2_3over2);
  const float Hj21 = px * (vy*px - vx*py) / enforceNotZero(px2_py2_3over2);

  // Row 1: Partial derivatives of rho wrt px, py, vx, vy.
  // Row 2: Partial derivatives of phi wrt px, py, vx, vy.
  // Row 3: Partial derivatives of rho_dot wrt px, py, vx, vy.
  *jacobian <<
      Hj00, Hj01, 0, 0,
      Hj10, Hj11, 0, 0,
      Hj20, Hj21, Hj00, Hj01;
}
