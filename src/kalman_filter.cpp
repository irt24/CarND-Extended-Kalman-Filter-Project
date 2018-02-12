#include "kalman_filter.h"
#include <assert.h>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
const float kSmallFloat = 0.0001;

float enforceNotZero(float x) {
  return fabs(x) < kSmallFloat ? kSmallFloat : x;
}
}  // end anonymous namespace

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  assert(z.size() == 2);
  UpdateInternal(z - H_ * x_);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  assert(z.size() == 3);
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // Map x_ to polar coordinates.
  float x_rho = sqrt(px*px + py*py);
  float x_phi = atan2(py, px);
  float x_rho_dot = (px*vx + py*vy) / enforceNotZero(x_rho);

  VectorXd h(3);
  h << x_rho, x_phi, x_rho_dot;

  // Make sure y_phi is in the interval [-pi/2, pi/2].
  VectorXd y = z - h;
  if (y(1) < -M_PI / 2) {
    y(1) += 2 * M_PI;
  } else if (y(1) > M_PI / 2) {
    y(1) -= 2 * M_PI;
  }

  UpdateInternal(y);
}

void KalmanFilter::UpdateInternal(const VectorXd& y) {
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K =  P_ * H_.transpose() * S.inverse();
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}
