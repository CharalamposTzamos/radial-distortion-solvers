#pragma once

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class Radial_12Point : public SolverEngine
			{
			public:
				double loss_scale;
				
				Radial_12Point()
				{
				}

				~Radial_12Point()
				{
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation.
				static constexpr bool needsGravity()
				{
					return false;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 12;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 4;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool Radial_12Point::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
                if (sample_ == nullptr)
					sample_number_ = data_.rows;

                Eigen::MatrixXd coefficients(sample_number_, 16);

                size_t pointIdx;
				double weight = 1.0;
                for (size_t i=0; i<sample_number_; ++i)
				{
                    if (sample_ == nullptr)
					{
						pointIdx = i;
						if (weights_ != nullptr)
							weight = weights_[i];
					}else{
						pointIdx = sample_[i];
						if (weights_ != nullptr)
							weight = weights_[sample_[i]];
					}
						

                    double u1 = data_.at<double>(pointIdx, 0), v1 = data_.at<double>(pointIdx, 1),
                        u2 = data_.at<double>(pointIdx, 2), v2 = data_.at<double>(pointIdx, 3);
                    double uv1 = u1*u1 + v1*v1, uv2 = u2*u2 + v2*v2;

					if (weights_ == nullptr)
					{
						coefficients(i, 0) = u1*u2;
						coefficients(i, 1) = u2*v1;
						coefficients(i, 2) = u2;
						coefficients(i, 3) = u1*v2;
						coefficients(i, 4) = v1*v2;
						coefficients(i, 5) = v2;
						coefficients(i, 6) = u1;
						coefficients(i, 7) = v1;
						coefficients(i, 8) = 1;
						coefficients(i, 9) = uv1*u2;
						coefficients(i, 10) = uv1*v2;
						coefficients(i, 11) = uv1;
						coefficients(i, 12) = uv2*u1;
						coefficients(i, 13) = uv2*v1;
						coefficients(i, 14) = uv2;
						coefficients(i, 15) = uv1*uv2;
					}else{
						const double
							weight_times_u1 = weight*u1,
							weight_times_v1 = weight*v1,
							weight_times_u2 = weight*u2,
							weight_times_v2 = weight*v2;
						
						coefficients(i, 0) = weight_times_u1*u2;
						coefficients(i, 1) = weight_times_u2*v1;
						coefficients(i, 2) = weight_times_u2;
						coefficients(i, 3) = weight_times_u1*v2;
						coefficients(i, 4) = weight_times_v1*v2;
						coefficients(i, 5) = weight_times_v2;
						coefficients(i, 6) = weight_times_u1;
						coefficients(i, 7) = weight_times_v1;
						coefficients(i, 8) = weight;
						coefficients(i, 9) = weight_times_u2*uv1;
						coefficients(i, 10) = weight_times_v2*uv1;
						coefficients(i, 11) = weight*uv1;
						coefficients(i, 12) = weight_times_u1*uv2;
						coefficients(i, 13) = weight_times_v1*uv2;
						coefficients(i, 14) = weight*uv2;
						coefficients(i, 15) = weight*uv1*uv2;
					}
                }


				Eigen::MatrixXd D1(sample_number_, 12);
				//D1 = coefficients.block<sample_number_,12>(0,0);
				D1 = coefficients.block(0, 0, sample_number_, 12);
				Eigen::MatrixXd D2(sample_number_, 12);
				//D2 = coefficients.block<sample_number_,4>(0,12);
				D2 = coefficients.block(0, 12, sample_number_, 4);

				Eigen::MatrixXd D3 = Eigen::MatrixXd::Zero(sample_number_, 12);
				D3.col(6) = coefficients.col(12);
				D3.col(7) = coefficients.col(13);
				D3.col(8) = coefficients.col(14);
				D3.col(11) = coefficients.col(15);

				Eigen::MatrixXd M(12, 4);
				M = -D1.colPivHouseholderQr().solve(D2);

				Eigen::MatrixXd D = Eigen::MatrixXd::Zero(4, 4);
				D.row(0) = M.row(6);
				D.row(1) = M.row(7);
				D.row(2) = M.row(8);
				D.row(3) = M.row(11);

				Eigen::EigenSolver<Eigen::MatrixXd> es(D);
				const Eigen::VectorXcd& eigenvalues = es.eigenvalues();


                Eigen::Matrix<double, 12, 1> f;
				Eigen::MatrixXd C(sample_number_, 12);
				for (size_t i = 0; i < 4; i++) {
					// Only consider real solutions.
					if (eigenvalues(i).imag() != 0) {
						continue;
					}
					
					double lambda = 1.0 / eigenvalues(i).real(); // distortion parameter
					C = D1 + lambda*D3;

					Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullV);


					f = svd.matrixV().col(11);

					FundamentalMatrix model;
                    model.descriptor.resize(3, 4);
					model.descriptor << 
                        f[0], f[1], f[2], f[9]/f[2],
                        f[3], f[4], f[5], lambda,
                        f[6], f[7], f[8], 0;
					models_.push_back(model);
				}

                return true;
            }
		}
	}
}