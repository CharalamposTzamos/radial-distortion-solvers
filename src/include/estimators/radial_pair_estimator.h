#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <chrono>
#include <variant>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"
#include "utils.h"
#include <algorithm>

#include "relative_pose/bundle.h"

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a essential matrix between two images. A model_ estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model_ from a minimal sample
			class _NonMinimalSolverEngine, // The solver used for estimating the model_ from a non-minimal sample
			size_t noSampledDistortionParametersMinimal,
			size_t noSampledDistortionParametersNonMinimal, 
			size_t sampleRandomDistortionParameter, // 0: no random distortion parameter, 1: random distortion parameter, 2: different dist for each view
			size_t residualCode> // 0: Tangent Sampson distance, 1: Sampson distance in distorted space, 2: Sampson distance in undistorted space
			class RadialPairEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			using rdVariant = std::variant<double, std::pair<double, double>>;

			// Minimal solver engine used for estimating a model_ from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model_ from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

			double loss_scale;
			double minDistortionParameter = -2.0;
			double maxDistortionParameter = 0.0;
			double minRadialTol = -2.0;
			double maxRadialTol = 0.5;
			std::vector<rdVariant> radial_distortion_parameters;

		public:
			RadialPairEstimator(
				double loss_scale_ = 1.0):
				// Minimal solver engine used for estimating a model from a minimal sample
				minimal_solver(std::make_shared<_MinimalSolverEngine>()),
				// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>()),
                loss_scale(loss_scale_)
			{
				// noSampledDistortionParametersMinimal and noSampledDistortionParametersNonMinimal should be either (k, k) or (0, k) or (k, 0)
				//non_minimal_solver->loss_scale = loss_scale;
				if (noSampledDistortionParametersMinimal == 1 or noSampledDistortionParametersNonMinimal == 1)
					radial_distortion_parameters.push_back(0.0);
				else if (noSampledDistortionParametersMinimal == 2 or noSampledDistortionParametersNonMinimal == 2)
				{
					radial_distortion_parameters.push_back(0.0);
					radial_distortion_parameters.push_back(-0.6);
				}
				else if (noSampledDistortionParametersMinimal == 3 or noSampledDistortionParametersNonMinimal == 3)
				{
					radial_distortion_parameters.push_back(0.0);
					radial_distortion_parameters.push_back(-0.6);
					radial_distortion_parameters.push_back(-1.2);
				}
				else if (noSampledDistortionParametersMinimal == 9 or noSampledDistortionParametersNonMinimal == 9)
				{
					radial_distortion_parameters.push_back(std::make_pair(0.0, 0.0));
					radial_distortion_parameters.push_back(std::make_pair(0.0, -0.6));
					radial_distortion_parameters.push_back(std::make_pair(0.0, -1.2));

					radial_distortion_parameters.push_back(std::make_pair(-0.6, 0.0));
					radial_distortion_parameters.push_back(std::make_pair(-0.6, -0.6));
					radial_distortion_parameters.push_back(std::make_pair(-0.6, -1.2));

					radial_distortion_parameters.push_back(std::make_pair(-1.2, 0.0));
					radial_distortion_parameters.push_back(std::make_pair(-1.2, -0.6));
					radial_distortion_parameters.push_back(std::make_pair(-1.2, -1.2));
				}
					
			}
			~RadialPairEstimator() {}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

            static constexpr bool useRadialDistortion()
			{
				return true;
			}

			// The size of a sample_ when doing inner RANSAC on a non-minimal sample
			OLGA_INLINE size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			// Return the minimal solver
			const _MinimalSolverEngine *getMinimalSolver() const
			{
				return minimal_solver.get();
			}

			// Return a mutable minimal solver
			_MinimalSolverEngine *getMutableMinimalSolver()
			{
				return minimal_solver.get();
			}

			// Return the minimal solver
			const _NonMinimalSolverEngine *getNonMinimalSolver() const
			{
				return non_minimal_solver.get();
			}

			// Return a mutable minimal solver
			_NonMinimalSolverEngine *getMutableNonMinimalSolver()
			{
				return non_minimal_solver.get();
			}

			// Estimating the essential matrix from a minimal sample
			OLGA_INLINE bool estimateModel(const cv::Mat& data, // The data_ points
				const size_t *sample, // The selected sample_ which will be used for estimation
				std::vector<Model>* models) const // The estimated model_ parameters
			{
				constexpr size_t sample_size = sampleSize();

				if (noSampledDistortionParametersMinimal == 0)
				{
					// if (!minimal_solver->estimateModel(data,
					// 	sample,
					// 	sample_size,
					// 	*models,
					// 	nullptr))
					// 	return false;

					std::vector<Model> tempModels;
					if (!minimal_solver->estimateModel(data,
						sample,
						sample_size,
						tempModels,
						nullptr))
						return false;
					
					for (auto &model : tempModels)
					{
						if (model.descriptor(0, 3) > maxRadialTol || model.descriptor(1, 3) > maxRadialTol)
							continue;

						if (model.descriptor(0, 3) < minRadialTol || model.descriptor(1, 3) < minRadialTol)
							continue;
						
						models->push_back(model);
					}
				}
				else
				{
					cv::Mat points(sample_size, data.cols, data.type());
					for (size_t i = 0; i < sample_size; ++i) 
					if (sample == nullptr)
						data.row(i).copyTo(points.row(i));
					else
						data.row(sample[i]).copyTo(points.row(i));

					Eigen::MatrixXd pointsEigen;
					cv::cv2eigen(points, pointsEigen);
					Eigen::Matrix<double, 2, sample_size> view1_pts = pointsEigen.block(0, 0, sample_size, 2).transpose();
					Eigen::Matrix<double, 2, sample_size> view2_pts = pointsEigen.block(0, 2, sample_size, 2).transpose();

					std::random_device randDevice;
					std::mt19937 gen(randDevice());
					std::uniform_real_distribution<> dis(minDistortionParameter, maxDistortionParameter);
					for (auto &rd : radial_distortion_parameters)
					{
						double rd1, rd2;
						if (sampleRandomDistortionParameter == 0)
						{
							if (std::holds_alternative<double>(rd))
							{
								rd1 = rd2 = std::get<double>(rd);
							}
							else if (std::holds_alternative<std::pair<double, double>>(rd))
							{
								auto [first, second] = std::get<std::pair<double, double>>(rd);
								rd1 = first;
								rd2 = second;
							}
						}
						else
						{
							double tmp_rd = dis(gen);
							rd1 = tmp_rd;
							rd2 = tmp_rd;
						}
							

						Eigen::Matrix<double, 2, Eigen::Dynamic> view1_pts_undistorted, view2_pts_undistorted;

						//undistort the points
						gcransac::utils::undistort_1param_division_model(rd1, view1_pts, &view1_pts_undistorted);
						gcransac::utils::undistort_1param_division_model(rd2, view2_pts, &view2_pts_undistorted);

						// make a points x 4 cv mat from view1_pts_undistorted and view2_pts_undistorted
						cv::Mat undistorted_points(sample_size, 4, CV_64F);
						for (size_t i = 0; i < sample_size; ++i)
						{
							undistorted_points.at<double>(i, 0) = view1_pts_undistorted(0, i);
							undistorted_points.at<double>(i, 1) = view1_pts_undistorted(1, i);
							undistorted_points.at<double>(i, 2) = view2_pts_undistorted(0, i);
							undistorted_points.at<double>(i, 3) = view2_pts_undistorted(1, i);
						}
	
						std::vector<Model> tempModels;
						if (!minimal_solver->estimateModel(undistorted_points,
							nullptr,
							sample_size,
							tempModels,
							nullptr))
							return false;

						for (auto &model : tempModels)
						{
							Model FundamentalMatrixWithRadial;
							FundamentalMatrixWithRadial.descriptor.resize(3, 4);
							FundamentalMatrixWithRadial.descriptor.block<3, 3>(0, 0) = model.descriptor;
							FundamentalMatrixWithRadial.descriptor(0, 3) = rd1;
							FundamentalMatrixWithRadial.descriptor(1, 3) = rd2;
							models->push_back(FundamentalMatrixWithRadial);
							//std::cout << "model from minimal:\n" << FundamentalMatrixWithRadial.descriptor << std::endl;
						}
					}	
				}
				
                return models->size() > 0;
			}

			// The squared sampson distance between a point_ correspondence and an essential matrix
			OLGA_INLINE double sampsonDistance(const cv::Mat& point_,
				const Eigen::Matrix3d& descriptor_) const
			{
				const double squared_distance = squaredSampsonDistance(point_, descriptor_);
				return sqrt(squared_distance);
			}

			// The sampson distance between a point_ correspondence and an essential matrix
			OLGA_INLINE double squaredSampsonDistance(const cv::Mat& point_,
				const Eigen::Matrix3d& descriptor_) const
			{
				const double* s = point_.ptr<double>(0);
				const double 
					&x1 = *s,
					&y1 = *(s + 1),
					&x2 = *(s + 2),
					&y2 = *(s + 3);

				const double 
					&e11 = descriptor_(0, 0),
					&e12 = descriptor_(0, 1),
					&e13 = descriptor_(0, 2),
					&e21 = descriptor_(1, 0),
					&e22 = descriptor_(1, 1),
					&e23 = descriptor_(1, 2);

				double rxc = e11 * x2 + e21 * y2 + descriptor_(2, 0);
				double ryc = e12 * x2 + e22 * y2 + descriptor_(2, 1);
				double rwc = e13 * x2 + e23 * y2 + descriptor_(2, 2);
				double r = (x1 * rxc + y1 * ryc + rwc);
				double rx = e11 * x1 + e12 * y1 + e13;
				double ry = e21 * x1 + e22 * y1 + e23;

				return r * r /
					(rxc * rxc + ryc * ryc + rx * rx + ry * ry);
			}

			// The squared sampson distance between a point_ correspondence and an essential matrix
			OLGA_INLINE double tangentSampsonDistance(const cv::Mat& point_,
				const Eigen::Matrix<double, 3, 4>& descriptor_) const
			{
				const double squared_distance = squaredTangentSampsonDistance(point_, descriptor_);
				return sqrt(squared_distance);
			}

			// The sampson distance between a point_ correspondence and an essential matrix
			OLGA_INLINE double squaredTangentSampsonDistance(const cv::Mat& point_,
				const Eigen::Matrix<double, 3, 4>& descriptor_) const
			{
				const double *s = point_.ptr<double>(0);

                double k1 = 0, k2 = 0; // The radial distortion parameter of the division model
				// If the descriptor has more than 3 columns, the radial distortion is estimated

				k1 = descriptor_(0, 3);
				k2 = descriptor_(1, 3);

				Eigen::Matrix3d F;
				F = descriptor_.block<3, 3>(0, 0);
                
                const double 
					&x1 = *s,
					&y1 = *(s + 1),
					&x2 = *(s + 2),
					&y2 = *(s + 3);

				Eigen::Matrix<double, 3, 2> dd1dx1, dd2dx2;
				dd1dx1 << 1.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    			dd2dx2 << 1.0, 0.0, 0.0, 1.0, 0.0, 0.0;

				Eigen::Matrix<double, 3, 1> d1, d2, h1, h2;

				d1 << x1, y1, 1 + k1 * (x1*x1 + y1*y1);
        		d2 << x2, y2, 1 + k2 * (x2*x2 + y2*y2);

				h1 = d1.normalized();
        		h2 = d2.normalized();

				double num = d2.transpose() * (F * d1);
        		num *= num;

				dd1dx1(2, 0) = 2 * k1 * x1;
				dd1dx1(2, 1) = 2 * k1 * y1;
				dd2dx2(2, 0) = 2 * k2 * x2;
				dd2dx2(2, 1) = 2 * k2 * y2;

				Eigen::Matrix<double, 3, 2> J1, J2;
				J1 = (Eigen::Matrix3d::Identity() - h1 * h1.transpose()) * dd1dx1 / d1.norm();
				J2 = (Eigen::Matrix3d::Identity() - h2 * h2.transpose()) * dd2dx2 / d2.norm();

				double den = (d2.transpose() * F * J1).squaredNorm() + (d1.transpose() * F.transpose() * J2).squaredNorm();
				double r2 = num / den;

				return r2;
			}

			// The squared residual function used for deciding which points are inliers
			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Model& model_) const
			{
				return squaredResidual(point_, model_.descriptor);
			}

			// The squared residual function used for deciding which points are inliers
			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				size_t rows = descriptor_.rows();
				size_t cols = descriptor_.cols();

				if(rows == 3 && cols == 4)
				{
					if(residualCode == 0)
					{
						return squaredTangentSampsonDistance(point_, descriptor_);
					}
					else if(residualCode == 1)
					{
						return squaredSampsonDistance(point_, descriptor_.block<3, 3>(0, 0));
					}
					else if(residualCode == 2)
					{
						//need to undisort the points here
						cv::Mat point(1, point_.cols, point_.type());
						point_.copyTo(point);

						Eigen::MatrixXd pointEigen;
						cv::cv2eigen(point, pointEigen);

						Eigen::Matrix<double, 2, 1> view1_pt = pointEigen.block(0, 0, 1, 2).transpose();
						Eigen::Matrix<double, 2, 1> view2_pt = pointEigen.block(0, 2, 1, 2).transpose();

						double k1 = descriptor_(0, 3);
						double k2 = descriptor_(1, 3);

						Eigen::Matrix<double, 2, Eigen::Dynamic> view1_pt_undistorted, view2_pt_undistorted;

						gcransac::utils::undistort_1param_division_model(k1, view1_pt, &view1_pt_undistorted);
						gcransac::utils::undistort_1param_division_model(k2, view2_pt, &view2_pt_undistorted);

						cv::Mat undistorted_point(1, 4, CV_64F);
						undistorted_point.at<double>(0, 0) = view1_pt_undistorted(0);
						undistorted_point.at<double>(0, 1) = view1_pt_undistorted(1);
						undistorted_point.at<double>(0, 2) = view2_pt_undistorted(0);
						undistorted_point.at<double>(0, 3) = view2_pt_undistorted(1);

						return squaredSampsonDistance(undistorted_point, descriptor_.block<3, 3>(0, 0));
					}	
				}
				else
				{
					//std::cerr << "The descriptor contains just E/F." << std::endl;
					//std::cout << "it shouldnt be here" << std::endl;
					return squaredSampsonDistance(point_, descriptor_);
				}
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				size_t rows = descriptor_.rows();
				size_t cols = descriptor_.cols();
				
				if(rows == 3 && cols == 4)
				{
					if(residualCode == 0)
					{
						return tangentSampsonDistance(point_, descriptor_);
					}
					else if(residualCode == 1)
					{
						return sampsonDistance(point_, descriptor_.block<3, 3>(0, 0));
					}
					else if(residualCode == 2)
					{
						//need to undisort the points here
						cv::Mat point(1, point_.cols, point_.type());
						point_.copyTo(point);

						Eigen::MatrixXd pointEigen;
						cv::cv2eigen(point, pointEigen);

						Eigen::Matrix<double, 2, 1> view1_pt = pointEigen.block(0, 0, 1, 2).transpose();
						Eigen::Matrix<double, 2, 1> view2_pt = pointEigen.block(0, 2, 1, 2).transpose();

						double k1 = descriptor_(0, 3);
						double k2 = descriptor_(1, 3);

						Eigen::Matrix<double, 2, Eigen::Dynamic> view1_pt_undistorted, view2_pt_undistorted;

						gcransac::utils::undistort_1param_division_model(k1, view1_pt, &view1_pt_undistorted);
						gcransac::utils::undistort_1param_division_model(k2, view2_pt, &view2_pt_undistorted);

						cv::Mat undistorted_point(1, 4, CV_64F);
						undistorted_point.at<double>(0, 0) = view1_pt_undistorted(0);
						undistorted_point.at<double>(0, 1) = view1_pt_undistorted(1);
						undistorted_point.at<double>(0, 2) = view2_pt_undistorted(0);
						undistorted_point.at<double>(0, 3) = view2_pt_undistorted(1);

						return sampsonDistance(undistorted_point, descriptor_.block<3, 3>(0, 0));
					}
				}
				else
				{
					//std::cout << "it shouldnt be here" << std::endl;
					return sampsonDistance(point_, descriptor_);
				}
			}

			// Validate the model by checking the number of inlier with symmetric epipolar distance
			// instead of Sampson distance. In general, Sampson distance is more accurate but less
			// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
			// every so-far-the-best model is checked if it has enough inlier with symmetric
			// epipolar distance as well. 
			bool isValidModel(Model& model_,
				const cv::Mat& data_,
				const std::vector<size_t> &inliers_,
				const size_t *minimal_sample_,
				const double threshold_,
				bool &model_updated_) const
			{
				return true;
			}
			
			// Estimating the model from a non-minimal sample
			OLGA_INLINE bool estimateModelNonminimal(
				const cv::Mat& data_,
				const size_t *sample_,
				const size_t &sample_number_,
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const
			{
				if (sample_number_ < nonMinimalSampleSize())
					return false;
				
				size_t sample_size = sample_number_;
				if(sample_ == nullptr)
					sample_size = data_.rows;
					
				if (noSampledDistortionParametersNonMinimal == 0)
				{
					// if (!non_minimal_solver->estimateModel(data_,
					// 	sample_,
					// 	sample_number_,
					// 	*models_,
					// 	weights_))
					// 	return false;

					std::vector<Model> tempModels;
					if (!non_minimal_solver->estimateModel(data_,
						sample_,
						sample_number_,
						tempModels,
						weights_))
						return false;
					
					for (auto &model : tempModels)
					{
						if (model.descriptor(0, 3) > maxRadialTol || model.descriptor(1, 3) > maxRadialTol)
							continue;

						if (model.descriptor(0, 3) < minRadialTol || model.descriptor(1, 3) < minRadialTol)
							continue;
						
						models_->push_back(model);
					}
				}
				else
				{
					cv::Mat points(sample_size, data_.cols, data_.type());
					for (size_t i = 0; i < sample_size; ++i) 
					if (sample_ == nullptr)
						data_.row(i).copyTo(points.row(i));
					else
						data_.row(sample_[i]).copyTo(points.row(i));

					Eigen::MatrixXd pointsEigen;
					cv::cv2eigen(points, pointsEigen);
					Eigen::Matrix<double, 2, Eigen::Dynamic> view1_pts = pointsEigen.block(0, 0, sample_size, 2).transpose();
					Eigen::Matrix<double, 2, Eigen::Dynamic> view2_pts = pointsEigen.block(0, 2, sample_size, 2).transpose();
					
					std::random_device randDevice;
    				std::mt19937 gen(randDevice());
					std::uniform_real_distribution<> dis(minDistortionParameter, maxDistortionParameter);
					for (auto &rd : radial_distortion_parameters)
					{
						double rd1, rd2;
						if (sampleRandomDistortionParameter == 0)
						{
							if (std::holds_alternative<double>(rd))
							{
								rd1 = rd2 = std::get<double>(rd);
							}
							else if (std::holds_alternative<std::pair<double, double>>(rd))
							{
								auto [first, second] = std::get<std::pair<double, double>>(rd);
								rd1 = first;
								rd2 = second;
							}
						}
						else
						{
							rd1 = dis(gen);
							rd2 = dis(gen);
						}
							

						Eigen::Matrix<double, 2, Eigen::Dynamic> view1_pts_undistorted, view2_pts_undistorted;
						//undistort the points

						gcransac::utils::undistort_1param_division_model(rd1, view1_pts, &view1_pts_undistorted);
						gcransac::utils::undistort_1param_division_model(rd2, view2_pts, &view2_pts_undistorted);

						// make a points x 4 cv mat from view1_pts_undistorted and view2_pts_undistorted
						cv::Mat undistorted_points(sample_size, 4, CV_64F);
						for (size_t i = 0; i < sample_size; ++i)
						{
							undistorted_points.at<double>(i, 0) = view1_pts_undistorted(0, i);
							undistorted_points.at<double>(i, 1) = view1_pts_undistorted(1, i);
							undistorted_points.at<double>(i, 2) = view2_pts_undistorted(0, i);
							undistorted_points.at<double>(i, 3) = view2_pts_undistorted(1, i);
						}

						std::vector<Model> tempModels;
						if (!non_minimal_solver->estimateModel(undistorted_points,
							nullptr,
							sample_size,
							tempModels,
							weights_))
							return false;

						for (auto &model : tempModels)
						{
							Model FundamentalMatrixWithRadial;
							FundamentalMatrixWithRadial.descriptor.resize(3, 4);
							FundamentalMatrixWithRadial.descriptor.block<3, 3>(0, 0) = model.descriptor;
							FundamentalMatrixWithRadial.descriptor(0, 3) = rd1;
							FundamentalMatrixWithRadial.descriptor(1, 3) = rd2;
							models_->push_back(FundamentalMatrixWithRadial);
							//std::cout << "model from minimal:\n" << FundamentalMatrixWithRadial.descriptor << std::endl;
						}
					}
				}
				return models_->size() > 0;
			}
		};
	}
}