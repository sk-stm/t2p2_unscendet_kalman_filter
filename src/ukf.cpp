#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // x, y, v, yaw, yaw_d
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_,n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/5.;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // + a, yaw_dd
  n_aug_ = n_x_ + 2;
  no_sigma_pts_aug_ = 2 * n_aug_ + 1;

  //set weights
  weights_ = VectorXd(no_sigma_pts_aug_);
  weights_.fill(0.5 / (lambda_+n_aug_));
  // first one gets more weight since it's the mean value
  weights_(0) = lambda_ / (lambda_+n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // ignore certain measurements for debugging
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR and not use_radar_) {
    return;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER and not use_laser_) {
    return;
  }

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    Initialize(meas_package);
    return;
  }


  // maintain timing
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // predict
  Prediction(dt);


  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else {
    throw runtime_error("Unsupported Sensor!");
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

void UKF::Initialize(MeasurementPackage const & meas_package) {

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    if (not use_radar_) return;

    // initial state (convert radar from polar to cartesian coordinates)
    x_(0) = cos(meas_package.raw_measurements_(1)) * meas_package.raw_measurements_(0);
    x_(1) = sin(meas_package.raw_measurements_(1)) * meas_package.raw_measurements_(0);

    // initial covariance of state
    P_.diagonal() << 0.5, 0.5, 1, 1, 1;

  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    if (not use_laser_) return;

    // initial state (copy coordinates)
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);
    
    // initial covariance of state
    P_.diagonal() << 0.2, 0.2, 1, 1, 1;
  }

  // done initializing, no need to predict or update
  is_initialized_ = true;
  previous_timestamp_ = meas_package.timestamp_;
  return;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //create augmented mean state
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(5) = x_;

  //create augmented covariance matrix
  MatrixXd P_aug = MatrixXd::Identity(n_aug_, n_aug_);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, no_sigma_pts_aug_);
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  //predict sigma points
  Xsig_pred_ = MatrixXd(x_.size(), no_sigma_pts_aug_);

  for (int i = 0; i < Xsig_aug.cols(); i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //predict state mean
  x_ = Xsig_pred_ * weights_;

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < no_sigma_pts_aug_; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // p_x, p_y
  int n_z = 2;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, no_sigma_pts_aug_);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for(int i = 0; i < Xsig_pred_.cols(); i++) {
    VectorXd col = Xsig_pred_.col(i);
    // just copy p_x and p_y
    Zsig.col(i) << col(0), col(1);
  }

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R.diagonal() << std_laspx_*std_laspx_, std_laspy_*std_laspy_;
  S = Zsig.colwise() - z_pred;
  S = MatrixXd(S.array().rowwise() * weights_.transpose().array()) * S.transpose();
  S += R;

  //calculate cross correlation matrix
  MatrixXd Tc = Xsig_pred_.colwise() - x_;
  Tc = MatrixXd(Tc.array().rowwise() * weights_.transpose().array());
  Tc = Tc * Zsig.transpose();
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  x_ += K * (meas_package.raw_measurements_ - z_pred);
  P_ -= K * S * K.transpose();

  // calculate NIS
  VectorXd diff = meas_package.raw_measurements_ - z_pred;
  NIS_laser_ = diff.transpose() * S.inverse() * diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // rho, phi, rho_d
  int n_z = 3;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, no_sigma_pts_aug_);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for(int i = 0; i < Xsig_pred_.cols(); i++) {
    VectorXd col = Xsig_pred_.col(i);
    double distance = sqrt(col(0)*col(0) + col(1)*col(1));
    double rho = distance;
    double phi = NormalizeAngle(atan2(col(1), col(0)));
    double rho_d = (col(0) * cos(col(3)) * col(2) + col(1) * sin(col(3)) * col(2)) / distance;
    
    Zsig.col(i) << rho, phi, rho_d;
  }

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd::Zero(n_z,n_z);
  R.diagonal() << std_radr_*std_radr_, std_radphi_*std_radphi_, std_radrd_*std_radrd_;
  S = Zsig.colwise() - z_pred;
  S = MatrixXd(S.array().rowwise() * weights_.transpose().array()) * S.transpose();
  S += R;

  //calculate cross correlation matrix
  MatrixXd Tc = Xsig_pred_.colwise() - x_;
  Tc = MatrixXd(Tc.array().rowwise() * weights_.transpose().array());
  Tc = Tc * Zsig.transpose();
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  x_ += K * (meas_package.raw_measurements_ - z_pred);
  P_ -= K * S * K.transpose();

  // calculate NIS
  VectorXd diff = meas_package.raw_measurements_ - z_pred;
  NIS_radar_ = diff.transpose() * S.inverse() * diff;
}
