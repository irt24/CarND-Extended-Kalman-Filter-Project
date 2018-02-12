#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <assert.h>
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // Initialize measurement covariance matrices (provided by manufacturer).
  R_laser_ = MatrixXd(2, 2);
  R_laser_ <<
      0.0225, 0,
      0, 0.0225;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ <<
      0.09, 0, 0,
      0, 0.0009, 0,
      0, 0, 0.09;

  // For measurement updates with lidar, H_laser_ is used to calculate y, S, K and P.
  // It is a projection from 4D state (x) to 2D state (z), where:
  // x = (px, py, vx, vy).transpose() and z = (px, py).transpose().
  H_laser_ = MatrixXd(2, 4);
  H_laser_ <<
      1, 0, 0, 0,
      0, 1, 0, 0;

  // For measurement updates with radar, the H matrix is a Jacobian (first order derivative of state).
  // Since it is a function of state, it's not initialized here, but rather computed on the fly.
  H_radar_ = MatrixXd(3, 4);

  // State covariance matrix.
  // This value is taken from the course, but I didn't find any explanation about where it comes from.
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ <<
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1000, 0,
      0, 0, 0, 1000;

  ekf_.x_ = VectorXd(4);
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.Q_ = MatrixXd(4, 4);
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/

  if (!is_initialized_) {
    float px, py, vx, vy;
    switch (measurement_pack.sensor_type_) {
      case MeasurementPackage::RADAR: {
        float rho = measurement_pack.raw_measurements_[0];
        float phi = measurement_pack.raw_measurements_[1];
        float rho_dot = measurement_pack.raw_measurements_[2];
        // Convert polar coordinates to cartesian coordinates.
        px = rho * cos(phi);
        py = rho * sin(phi);
        vx = rho_dot * cos(phi);
        vy = rho_dot * sin(phi);
        break;
      } case MeasurementPackage::LASER: {
        px = measurement_pack.raw_measurements_[0];
        py = measurement_pack.raw_measurements_[1];
        vx = vy = 0;  // Laser doesn't measure speed directly.
        break;
      } default: {
        assert(false);  // Unrecognized sensor type.
      }
    }

    ekf_.x_ << px, py, vx, vy;
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // Update the state transition matrix F according to the elapsed time.
  const float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;  // microseconds to seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_ <<
      1, 0, dt, 0,
      0, 1, 0, dt,
      0, 0, 1, 0,
      0, 0, 0, 1;

  // Update the process noise covariance matrix.
  const float noise_ax_sq = 9 * 9;
  const float noise_ay_sq = 9 * 9;
  const float dt2 = dt * dt;
  const float dt3_2 = dt2 * dt / 2;
  const float dt4_4 = dt3_2 * dt / 2;

  ekf_.Q_ <<
      dt4_4 * noise_ax_sq, 0, dt3_2 * noise_ax_sq, 0,
      0, dt4_4 * noise_ay_sq, 0, dt3_2 * noise_ay_sq,
      dt3_2 * noise_ax_sq, 0, dt2 * noise_ax_sq, 0,
      0, dt3_2 * noise_ay_sq, 0, dt2 * noise_ay_sq;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  switch (measurement_pack.sensor_type_) {
    case MeasurementPackage::RADAR: {
      tools.CalculateJacobian(ekf_.x_, &H_radar_);
      ekf_.H_ = H_radar_;
      ekf_.R_ = R_radar_;
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);
      break;
    } case MeasurementPackage::LASER: {
      ekf_.H_ = H_laser_;
      ekf_.R_ = R_laser_;
      ekf_.Update(measurement_pack.raw_measurements_);
      break;
    } default: {
      assert(false);  // Unrecognized sensor type.
    }
  }

  // Print the output.
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
