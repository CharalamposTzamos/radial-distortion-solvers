// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include "solver_engine.h"
#include "fundamental_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class FundamentalMatrixPEPnineKFK : public SolverEngine
			{
			public:
				FundamentalMatrixPEPnineKFK()
				{
				}

				~FundamentalMatrixPEPnineKFK()
				{
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
					return 6;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 9;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation.
				static constexpr bool needsGravity()
				{
					return false;
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

			OLGA_INLINE bool FundamentalMatrixPEPnineKFK::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				Eigen::MatrixXd D1(sample_number_, 9);
				Eigen::MatrixXd D2(sample_number_, 9);
				D2.setZero();
				Eigen::MatrixXd D3(sample_number_, 9);
				D3.setZero();
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;

				// form a linear system: i-th row of A(=a) represents
				// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
				double weight = 1.0;
				size_t offset;
				for (size_t i = 0; i < sample_number_; i++)
				{
					if (sample_ == nullptr)
					{
						offset = cols * i;
						if (weights_ != nullptr)
							weight = weights_[i];
					} 
					else
					{
						offset = cols * sample_[i];
						if (weights_ != nullptr)
							weight = weights_[sample_[i]];
					}

					const double
						u1 = data_ptr[offset],
						v1 = data_ptr[offset + 1],
						u2 = data_ptr[offset + 2],
						v2 = data_ptr[offset + 3],
						uv1 = u1*u1+v1*v1,
						uv2 = u2*u2+v2*v2;

					// If not weighted least-squares is applied
                    if (true)
					{
                        D1(i, 0) = u1*u2;
                        D1(i, 1) = u2*v1;
                        D1(i, 2) = u2;
                        D1(i, 3) = u1*v2;
                        D1(i, 4) = v1*v2;
                        D1(i, 5) = v2;
                        D1(i, 6) = u1;
                        D1(i, 7) = v1;
                        D1(i, 8) = 1.0;

                        D2(i,2) = u2*uv1;
                        D2(i,5) = v2*uv1;
                        D2(i,6) = u1*uv2;
                        D2(i,7) = v1*uv2;
                        D2(i,8) = uv1+uv2;

                        D3(i,8) = uv1*uv2;
                    }
					/*else{
                        const double
							weight_times_u1 = weight*u1,
							weight_times_v1 = weight*v1,
							weight_times_u2 = weight*u2,
							weight_times_v2 = weight*v2;
                        
                        D1(i, 0) = weight_times_u1*u2;
                        D1(i, 1) = weight_times_u2*v1;
                        D1(i, 2) = weight_times_u2;
                        D1(i, 3) = weight_times_u1*v2;
                        D1(i, 4) = weight_times_v1*v2;
                        D1(i, 5) = weight_times_v2;
                        D1(i, 6) = weight_times_u1;
                        D1(i, 7) = weight_times_v1;
                        D1(i, 8) = weight;

                        D2(i,2) = weight_times_u2*uv1;
                        D2(i,5) = weight_times_v2*uv1;
                        D2(i,6) = weight_times_u1*uv2;
                        D2(i,7) = weight_times_v1*uv2;
                        D2(i,8) = weight*uv1 + weight*uv2;

                        D3(i,8) = weight*uv1*uv2;
                    }*/

				}
				Eigen::MatrixXd M(9, 6);
				Eigen::MatrixXd D4(sample_number_, 6);
				D4 << D2.col(2), D2.col(5), D2.col(6), D2.col(7), D2.col(8), D3.col(8);
				M = -D1.colPivHouseholderQr().solve(D4);

				Eigen::MatrixXd D = Eigen::MatrixXd::Zero(6, 6);
				D(0,5) = 1.0;
				D.row(1) = M.row(2);
				D.row(2) = M.row(5);
				D.row(3) = M.row(6);
				D.row(4) = M.row(7);
				D.row(5) = M.row(8);

				Eigen::EigenSolver<Eigen::MatrixXd> es(D);
				const Eigen::VectorXcd& eigenvalues = es.eigenvalues();
				Eigen::Matrix<double, 9, 1> f;
				Eigen::MatrixXd C(sample_number_, 9);
				for (size_t i = 0; i < 6; i++) {
					// Only consider real solutions.
					if (eigenvalues(i).imag() != 0) {
						continue;
					}
					
					double lambda = 1.0 / eigenvalues(i).real(); // distortion parameter
					C = D1 + lambda*D2 + lambda*lambda*D3;

					Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullV);


					f = svd.matrixV().col(8);

					FundamentalMatrix model;
                    model.descriptor.resize(3, 4);
					model.descriptor << 
                        f[0], f[1], f[2], lambda,
                        f[3], f[4], f[5], lambda,
                        f[6], f[7], f[8], 0;
					models_.push_back(model);
				}

				return true;
			}
		}
	}
}