#ifndef POSELIB_JACOBIAN_IMPL_H_
#define POSELIB_JACOBIAN_IMPL_H_

#include "essential.h"
#include "quaternion.h"

namespace pose_lib {

// from Viktor Kocur's code
class UniformWeightVector {
  public:
    UniformWeightVector() {}
    constexpr double operator[](std::size_t idx) const { return 1.0; }
};

// from Viktor Kocur's code
class UniformWeightVectors { // this corresponds to std::vector<std::vector<double>> used for generalized cameras etc
  public:
    UniformWeightVectors() {}
    constexpr const UniformWeightVector &operator[](std::size_t idx) const { return w; }
    const UniformWeightVector w;
    typedef UniformWeightVector value_type;
};

Eigen::Matrix3d skew(const Eigen::Vector3d &x){
    Eigen::Matrix3d s;
    s <<  0, -x(2), x(1), x(2), 0, -x(0),
        -x(1), x(0), 0;
    return s;
}

// from Viktor Kocur's code
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class ThreeViewRelativePoseJacobianAccumulator {
  public:
    ThreeViewRelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                             const std::vector<Point2D> &points2D_2,
                                             const std::vector<Point2D> &points2D_3,
                                             const LossFunction &l,
                                             const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), x3(points2D_3), loss_fn(l), weights(w) {}

    double residual(const ThreeViewCameraPose &three_view_pose) const {
        Eigen::Matrix3d E12, E13, E23;
        essential_from_motion(three_view_pose.pose12, &E12);
        essential_from_motion(three_view_pose.pose13, &E13);
        essential_from_motion(three_view_pose.pose23(), &E23);

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            // E12          
            double C12 = x2[k].homogeneous().dot(E12 * x1[k].homogeneous());
            double nJc12_sq = (E12.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E12.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();
            double r12_sq = (C12 * C12) / nJc12_sq;
            double loss12 = loss_fn.loss(r12_sq);
            if (loss12 == 0.0)
                continue;

            // E13            
            double C13 = x3[k].homogeneous().dot(E13 * x1[k].homogeneous());
            double nJc13_sq = (E13.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                              (E13.block<3, 2>(0, 0).transpose() * x3[k].homogeneous()).squaredNorm();
            
            double r13_sq = (C13 * C13) / nJc13_sq;
            double loss13 = loss_fn.loss(r13_sq);
            if (loss13 == 0.0)
                continue;
            
            // E23            
            double C23 = x3[k].homogeneous().dot(E23 * x2[k].homogeneous());
            double nJc23_sq = (E23.block<2, 3>(0, 0) * x2[k].homogeneous()).squaredNorm() +
                              (E23.block<3, 2>(0, 0).transpose() * x3[k].homogeneous()).squaredNorm();

            double r23_sq = (C23 * C23) / nJc23_sq;
            double loss23 = loss_fn.loss(r23_sq);
            if (loss23 == 0.0)
                continue;

            cost += weights[k] * loss12;
            cost += weights[k] * loss13;
            cost += weights[k] * loss23;
        }

        return cost;
    }

    size_t accumulate(const ThreeViewCameraPose &three_view_pose, Eigen::Matrix<double, 11, 11> &JtJ,
                      Eigen::Matrix<double, 11, 1> &Jtr) {
        // We use tangent bases for t12 and direct vector for t23
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        
        CameraPose pose12 = three_view_pose.pose12;
        
        if (std::abs(pose12.t.x()) < std::abs(pose12.t.y())) {
            // x < y
            if (std::abs(pose12.t.x()) < std::abs(pose12.t.z())) {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose12.t.y()) < std::abs(pose12.t.z())) {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose12.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose12.t).normalized();

        Eigen::Matrix3d E12, R12, E13, R13, E23, R23;

        R12 = pose12.R();
        essential_from_motion(pose12, &E12);
        
        R13 = three_view_pose.pose13.R();
        essential_from_motion(three_view_pose.pose13, &E13);

        CameraPose pose23 = three_view_pose.pose23();
        R23 = pose23.R();
        essential_from_motion(pose23, &E23);
                
        // Matrices contain the jacobians of E12 w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dE12dr12;
        Eigen::Matrix<double, 9, 2> dE12dt12;

        // Each column is vec(E12*skew(e_k)) where e_k is k:th basis vector
        dE12dr12.block<3, 1>(0, 0).setZero();
        dE12dr12.block<3, 1>(0, 1) = -E12.col(2);
        dE12dr12.block<3, 1>(0, 2) = E12.col(1);
        dE12dr12.block<3, 1>(3, 0) = E12.col(2);
        dE12dr12.block<3, 1>(3, 1).setZero();
        dE12dr12.block<3, 1>(3, 2) = -E12.col(0);
        dE12dr12.block<3, 1>(6, 0) = -E12.col(1);
        dE12dr12.block<3, 1>(6, 1) = E12.col(0);
        dE12dr12.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R12)
        dE12dt12.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R12.col(0));
        dE12dt12.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R12.col(0));
        dE12dt12.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R12.col(1));
        dE12dt12.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R12.col(1));
        dE12dt12.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R12.col(2));
        dE12dt12.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R12.col(2));
        
        // Matrices contain the jacobians of E12 w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dE13dr13;
        Eigen::Matrix<double, 9, 3> dE13dt13;
        
        // Each column is vec(E13*skew(e_k)) where e_k is k:th basis vector
        dE13dr13.block<3, 1>(0, 0).setZero();
        dE13dr13.block<3, 1>(0, 1) = -E13.col(2);
        dE13dr13.block<3, 1>(0, 2) = E13.col(1);
        dE13dr13.block<3, 1>(3, 0) = E13.col(2);
        dE13dr13.block<3, 1>(3, 1).setZero();
        dE13dr13.block<3, 1>(3, 2) = -E13.col(0);
        dE13dr13.block<3, 1>(6, 0) = -E13.col(1);
        dE13dr13.block<3, 1>(6, 1) = E13.col(0);
        dE13dr13.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(e_k)*R13) where e_k is k:th basis vector
        Eigen::Matrix3d dE13dt13_0, dE13dt13_1, dE13dt13_2;
        dE13dt13_0.row(0).setZero();
        dE13dt13_0.row(1) = -R13.row(2);
        dE13dt13_0.row(2) = R13.row(1);

        dE13dt13_1.row(0) = R13.row(2);
        dE13dt13_1.row(1).setZero();
        dE13dt13_1.row(2) = - R13.row(0);

        dE13dt13_2.row(0) = - R13.row(1);
        dE13dt13_2.row(1) = R13.row(0);
        dE13dt13_2.row(2).setZero();

        dE13dt13.col(0) = Eigen::Map<Eigen::VectorXd>(dE13dt13_0.data(), dE13dt13_0.size());
        dE13dt13.col(1) = Eigen::Map<Eigen::VectorXd>(dE13dt13_1.data(), dE13dt13_1.size());
        dE13dt13.col(2) = Eigen::Map<Eigen::VectorXd>(dE13dt13_2.data(), dE13dt13_2.size());

        //TODO: this part calculates dE23dX and is not optimized yet

        // define skew(e_k)
        Eigen::Matrix3d b_0 = skew(Eigen::Vector3d::UnitX());
        Eigen::Matrix3d b_1 = skew(Eigen::Vector3d::UnitY());
        Eigen::Matrix3d b_2 = skew(Eigen::Vector3d::UnitZ());

        Eigen::Matrix3d dE23dr12_0, dE23dr12_1, dE23dr12_2;
        dE23dr12_0 = skew(R13 * b_0 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_0 * R12.transpose();
        dE23dr12_1 = skew(R13 * b_1 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_1 * R12.transpose();
        dE23dr12_2 = skew(R13 * b_2 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_2 * R12.transpose();

        Eigen::Matrix3d dE23dr13_0, dE23dr13_1, dE23dr13_2;
        dE23dr13_0 = - dE23dr12_0;
        dE23dr13_1 = - dE23dr12_1;
        dE23dr13_2 = - dE23dr12_2;

        // dE23dt12 = skew(tangent_basis_k) * R23
        Eigen::Matrix3d dE23dt12_0, dE23dt12_1;
        dE23dt12_0 = - skew(R23 * tangent_basis.col(0)) * R23;
        dE23dt12_1 = - skew(R23 * tangent_basis.col(1)) * R23;

        // dE23dt13 = skew(e_k) * R23
        Eigen::Matrix3d dE23dt13_0, dE23dt13_1, dE23dt13_2;
        dE23dt13_0.row(0).setZero();
        dE23dt13_0.row(1) = -R23.row(2);
        dE23dt13_0.row(2) = R23.row(1);

        dE23dt13_1.row(0) = R23.row(2);
        dE23dt13_1.row(1).setZero();
        dE23dt13_1.row(2) = - R23.row(0);

        dE23dt13_2.row(0) = - R23.row(1);
        dE23dt13_2.row(1) = R23.row(0);
        dE23dt13_2.row(2).setZero();

        Eigen::Matrix<double, 9, 3> dE23dr12;
        Eigen::Matrix<double, 9, 3> dE23dr13;
        Eigen::Matrix<double, 9, 2> dE23dt12;
        Eigen::Matrix<double, 9, 3> dE23dt13;

        dE23dr12.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dr12_0.data(), dE23dr12_0.size());
        dE23dr12.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dr12_1.data(), dE23dr12_1.size());
        dE23dr12.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dr12_2.data(), dE23dr12_2.size());

        dE23dr13.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dr13_0.data(), dE23dr13_0.size());
        dE23dr13.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dr13_1.data(), dE23dr13_1.size());
        dE23dr13.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dr13_2.data(), dE23dr13_2.size());

        dE23dt12.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dt12_0.data(), dE23dt12_0.size());
        dE23dt12.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dt12_1.data(), dE23dt12_1.size());

        dE23dt13.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dt13_0.data(), dE23dt13_0.size());
        dE23dt13.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dt13_1.data(), dE23dt13_1.size());
        dE23dt13.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dt13_2.data(), dE23dt13_2.size());


        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C12 = x2[k].homogeneous().dot(E12 * x1[k].homogeneous());
            double C13 = x3[k].homogeneous().dot(E13 * x1[k].homogeneous());
            double C23 = x3[k].homogeneous().dot(E23 * x2[k].homogeneous());

            // J_C12 is the Jacobian of the epipolar constraint w.r12.t. the image points
            Eigen::Vector4d J_C12;
            J_C12 << E12.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), E12.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C12 = J_C12.norm();
            const double inv_nJ_C12 = 1.0 / nJ_C12;
            const double r12 = C12 * inv_nJ_C12;
            const double weight12 = weights[k] * loss_fn.weight(r12 * r12);
            if (weight12 == 0.0) {
                continue;
            }
                        
            Eigen::Vector4d J_C13;
            J_C13 << E13.block<3, 2>(0, 0).transpose() * x3[k].homogeneous(), E13.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C13 = J_C13.norm();
            const double inv_nJ_C13 = 1.0 / nJ_C13;
            const double r13 = C13 * inv_nJ_C13;

            const double weight13 = weights[k] * loss_fn.weight(r13 * r13);
            if (weight13 == 0.0) {
                continue;
            }
            
            Eigen::Vector4d J_C23;
            J_C23 << E23.block<3, 2>(0, 0).transpose() * x3[k].homogeneous(), E23.block<2, 3>(0, 0) * x2[k].homogeneous();
            const double nJ_C23 = J_C23.norm();
            const double inv_nJ_C23 = 1.0 / nJ_C23;
            const double r23 = C23 * inv_nJ_C23;

            const double weight23 = weights[k] * loss_fn.weight(r23 * r23);
            if (weight23 == 0.0) {
                continue;
            }
                        
            num_residuals += 3;

            // Compute Jacobian of Sampson error w.r12.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dSdE12;
            dSdE12 << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s12 = C12 * inv_nJ_C12 * inv_nJ_C12;
            dSdE12(0) -= s12 * (J_C12(2) * x1[k](0) + J_C12(0) * x2[k](0));
            dSdE12(1) -= s12 * (J_C12(3) * x1[k](0) + J_C12(0) * x2[k](1));
            dSdE12(2) -= s12 * (J_C12(0));
            dSdE12(3) -= s12 * (J_C12(2) * x1[k](1) + J_C12(1) * x2[k](0));
            dSdE12(4) -= s12 * (J_C12(3) * x1[k](1) + J_C12(1) * x2[k](1));
            dSdE12(5) -= s12 * (J_C12(1));
            dSdE12(6) -= s12 * (J_C12(2));
            dSdE12(7) -= s12 * (J_C12(3));
            dSdE12 *= inv_nJ_C12;

            Eigen::Matrix<double, 1, 9> dSdE13;
            dSdE13 << x1[k](0) * x3[k](0), x1[k](0) * x3[k](1), x1[k](0), x1[k](1) * x3[k](0), x1[k](1) * x3[k](1),
                x1[k](1), x3[k](0), x3[k](1), 1.0;
            const double s13 = C13 * inv_nJ_C13 * inv_nJ_C13;
            dSdE13(0) -= s13 * (J_C13(2) * x1[k](0) + J_C13(0) * x3[k](0));
            dSdE13(1) -= s13 * (J_C13(3) * x1[k](0) + J_C13(0) * x3[k](1));
            dSdE13(2) -= s13 * (J_C13(0));
            dSdE13(3) -= s13 * (J_C13(2) * x1[k](1) + J_C13(1) * x3[k](0));
            dSdE13(4) -= s13 * (J_C13(3) * x1[k](1) + J_C13(1) * x3[k](1));
            dSdE13(5) -= s13 * (J_C13(1));
            dSdE13(6) -= s13 * (J_C13(2));
            dSdE13(7) -= s13 * (J_C13(3));
            dSdE13 *= inv_nJ_C13;
            
            Eigen::Matrix<double, 1, 9> dSdE23;
            dSdE23 << x2[k](0) * x3[k](0), x2[k](0) * x3[k](1), x2[k](0), x2[k](1) * x3[k](0), x2[k](1) * x3[k](1),
                x2[k](1), x3[k](0), x3[k](1), 1.0;
            const double s23 = C23 * inv_nJ_C23 * inv_nJ_C23;
            dSdE23(0) -= s23 * (J_C23(2) * x2[k](0) + J_C23(0) * x3[k](0));
            dSdE23(1) -= s23 * (J_C23(3) * x2[k](0) + J_C23(0) * x3[k](1));
            dSdE23(2) -= s23 * (J_C23(0));
            dSdE23(3) -= s23 * (J_C23(2) * x2[k](1) + J_C23(1) * x3[k](0));
            dSdE23(4) -= s23 * (J_C23(3) * x2[k](1) + J_C23(1) * x3[k](1));
            dSdE23(5) -= s23 * (J_C23(1));
            dSdE23(6) -= s23 * (J_C23(2));
            dSdE23(7) -= s23 * (J_C23(3));
            dSdE23 *= inv_nJ_C23;
            
            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 11> J12;
            J12.block<1, 3>(0, 0) = dSdE12 * dE12dr12;
            J12.block<1, 2>(0, 3) = dSdE12 * dE12dt12;
            J12.block<1, 6>(0, 6).setZero();

            Eigen::Matrix<double, 1, 11> J13;
            J13.block<1, 5>(0, 0).setZero();
            J13.block<1, 3>(0, 5) = dSdE13 * dE13dr13;
            J13.block<1, 3>(0, 8) = dSdE13 * dE13dt13;

            Eigen::Matrix<double, 1, 11> J23;
            J23.block<1, 3>(0, 0) = dSdE23 * dE23dr12;
            J23.block<1, 2>(0, 3) = dSdE23 * dE23dt12;
            J23.block<1, 3>(0, 5) = dSdE23 * dE23dr13;
            J23.block<1, 3>(0, 8) = dSdE23 * dE23dt13;


            // Accumulate into Jtr
            Jtr += weight12 * C12 * inv_nJ_C12 * J12.transpose();
            Jtr += weight13 * C13 * inv_nJ_C13 * J13.transpose();
            Jtr += weight23 * C23 * inv_nJ_C23 * J23.transpose();

            for (int row = 0; row < 11; row++)
                for (int col = 0; col < 11; col++)
                    if (row >= col) {
                        JtJ(row, col) += weight12 * (J12(row) * J12(col));
                        JtJ(row, col) += weight13 * (J13(row) * J13(col));
                        JtJ(row, col) += weight23 * (J23(row) * J23(col));
                    }
        }
        return num_residuals;
    }

    ThreeViewCameraPose step(Eigen::Matrix<double, 11, 1> dp, const ThreeViewCameraPose &three_view_pose) const {
        CameraPose pose12_new, pose13_new;

        pose12_new.q = quat_step_post(three_view_pose.pose12.q, dp.block<3, 1>(0, 0));
        pose12_new.t = three_view_pose.pose12.t + tangent_basis * dp.block<2, 1>(3, 0);

        pose13_new.q = quat_step_post(three_view_pose.pose13.q, dp.block<3, 1>(5, 0));
        pose13_new.t = three_view_pose.pose13.t + dp.block<3, 1>(8, 0);

        ThreeViewCameraPose three_view_pose_new(pose12_new, pose13_new);
        return three_view_pose_new;
    }
    typedef ThreeViewCameraPose param_t;
    static constexpr size_t num_params = 11;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};


template <typename CameraModel, typename LossFunction>
class CameraJacobianAccumulator 
{
    public:
        CameraJacobianAccumulator(
            const cv::Mat& correspondences_,
            const size_t* sample_,
            const size_t& sample_size_,
            const Camera &cam,
            const LossFunction &loss,
            const double *w = nullptr) : 
                correspondences(&correspondences_), 
                sample(sample_),
                sample_size(sample_size_),
                loss_fn(loss),
                camera(cam),
                weights(w) {}

        double residual(const CameraPose &pose) const 
        {
            double cost = 0;
            Eigen::Vector2d x;
            Eigen::Vector3d X;
            size_t rowIdx = 0;
            for (size_t i = 0; i < correspondences->rows; ++i) 
            {
                if (sample == nullptr)
                    rowIdx = i;
                else
                    rowIdx = sample[i];
                
                x(0) = correspondences->at<double>(rowIdx, 0);
                x(1) = correspondences->at<double>(rowIdx, 1);
                X(0) = correspondences->at<double>(rowIdx, 2);
                X(1) = correspondences->at<double>(rowIdx, 3);
                X(2) = correspondences->at<double>(rowIdx, 4);

                const Eigen::Vector3d Z = pose.apply(X);
                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization
                if (Z(2) < 0)
                    continue;
                const double inv_z = 1.0 / Z(2);
                Eigen::Vector2d p(Z(0) * inv_z, Z(1) * inv_z);
                CameraModel::project(camera.params, p, &p);
                const double r0 = p(0) - x(0);
                const double r1 = p(1) - x(1);
                const double r_squared = r0 * r0 + r1 * r1;
                cost += weights[i] * loss_fn.loss(r_squared);
            }
            return cost;
        }

        // computes J.transpose() * J and J.transpose() * res
        // Only computes the lower half of JtJ
        size_t accumulate(
            const CameraPose &pose, 
            Eigen::Matrix<double, 6, 6> &JtJ,
            Eigen::Matrix<double, 6, 1> &Jtr) const 
        {
            Eigen::Matrix3d R = pose.R();
            Eigen::Matrix2d Jcam;
            Jcam.setIdentity(); // we initialize to identity here (this is for the calibrated case)
            size_t num_residuals = 0;

            Eigen::Vector2d x;
            Eigen::Vector3d X;
            size_t rowIdx = 0;
            for (size_t i = 0; i < correspondences->rows; ++i) 
            {
                if (sample == nullptr)
                    rowIdx = i;
                else
                    rowIdx = sample[i];

                x(0) = correspondences->at<double>(rowIdx, 0);
                x(1) = correspondences->at<double>(rowIdx, 1);
                X(0) = correspondences->at<double>(rowIdx, 2);
                X(1) = correspondences->at<double>(rowIdx, 3);
                X(2) = correspondences->at<double>(rowIdx, 4);
            
                const Eigen::Vector3d Z = R * X + pose.t;
                const Eigen::Vector2d z = Z.hnormalized();

                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization
                if (Z(2) < 0)
                    continue;

                // Project with intrinsics
                Eigen::Vector2d zp = z;
                CameraModel::project_with_jac(camera.params, z, &zp, &Jcam);

                // Setup residual
                Eigen::Vector2d r = zp - x;
                const double r_squared = r.squaredNorm();
                const double weight = weights[i] * loss_fn.weight(r_squared);

                if (weight == 0.0) {
                    continue;
                }
                num_residuals++;

                // Compute jacobian w.r.t. Z (times R)
                Eigen::Matrix<double, 2, 3> dZ;
                dZ.block<2, 2>(0, 0) = Jcam;
                dZ.col(2) = -Jcam * z;
                dZ *= 1.0 / Z(2);
                dZ *= R;

                const double X0 = X(0);
                const double X1 = X(1);
                const double X2 = X(2);
                const double dZtdZ_0_0 = weight * dZ.col(0).dot(dZ.col(0));
                const double dZtdZ_1_0 = weight * dZ.col(1).dot(dZ.col(0));
                const double dZtdZ_1_1 = weight * dZ.col(1).dot(dZ.col(1));
                const double dZtdZ_2_0 = weight * dZ.col(2).dot(dZ.col(0));
                const double dZtdZ_2_1 = weight * dZ.col(2).dot(dZ.col(1));
                const double dZtdZ_2_2 = weight * dZ.col(2).dot(dZ.col(2));
                JtJ(0, 0) += X2 * (X2 * dZtdZ_1_1 - X1 * dZtdZ_2_1) + X1 * (X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1);
                JtJ(1, 0) += -X2 * (X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1) - X1 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
                JtJ(2, 0) += X1 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0) - X2 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
                JtJ(3, 0) += X1 * dZtdZ_2_0 - X2 * dZtdZ_1_0;
                JtJ(4, 0) += X1 * dZtdZ_2_1 - X2 * dZtdZ_1_1;
                JtJ(5, 0) += X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1;
                JtJ(1, 1) += X2 * (X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0) + X0 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
                JtJ(2, 1) += -X2 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) - X0 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0);
                JtJ(3, 1) += X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0;
                JtJ(4, 1) += X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1;
                JtJ(5, 1) += X2 * dZtdZ_2_0 - X0 * dZtdZ_2_2;
                JtJ(2, 2) += X1 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) + X0 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
                JtJ(3, 2) += X0 * dZtdZ_1_0 - X1 * dZtdZ_0_0;
                JtJ(4, 2) += X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0;
                JtJ(5, 2) += X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0;
                JtJ(3, 3) += dZtdZ_0_0;
                JtJ(4, 3) += dZtdZ_1_0;
                JtJ(5, 3) += dZtdZ_2_0;
                JtJ(4, 4) += dZtdZ_1_1;
                JtJ(5, 4) += dZtdZ_2_1;
                JtJ(5, 5) += dZtdZ_2_2;
                r *= weight;
                Jtr(0) += (r(0) * (X1 * dZ(0, 2) - X2 * dZ(0, 1)) + r(1) * (X1 * dZ(1, 2) - X2 * dZ(1, 1)));
                Jtr(1) += (-r(0) * (X0 * dZ(0, 2) - X2 * dZ(0, 0)) - r(1) * (X0 * dZ(1, 2) - X2 * dZ(1, 0)));
                Jtr(2) += (r(0) * (X0 * dZ(0, 1) - X1 * dZ(0, 0)) + r(1) * (X0 * dZ(1, 1) - X1 * dZ(1, 0)));
                Jtr(3) += (dZ(0, 0) * r(0) + dZ(1, 0) * r(1));
                Jtr(4) += (dZ(0, 1) * r(0) + dZ(1, 1) * r(1));
                Jtr(5) += (dZ(0, 2) * r(0) + dZ(1, 2) * r(1));
            }
            return num_residuals;
        }

        CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const 
        {
            CameraPose pose_new;
            // The rotation is parameterized via the lie-rep. and post-multiplication
            //   i.e. R(delta) = R * expm([delta]_x)
            pose_new.q = pose.quat_step_post(pose.q, dp.block<3, 1>(0, 0));

            // Translation is parameterized as (negative) shift in position
            //  i.e. t(delta) = t + R*delta
            pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
            return pose_new;
        }
        
        typedef CameraPose param_t;
        static constexpr size_t num_params = 6;

    private:
        const cv::Mat* correspondences;
        const size_t* sample;
        const size_t sample_size;

        const Camera &camera;
        const LossFunction &loss_fn;
        const double *weights;
};

// Non-linear refinement of transfer error |x2 - pi(H*x1)|^2, parameterized by fixing H(2,2) = 1
// I did some preliminary experiments comparing different error functions (e.g. symmetric and transfer)
// as well as other parameterizations (different affine patches, SVD as in Bartoli/Sturm, etc)
// but it does not seem to have a big impact (and is sometimes even worse)
// Implementations of these can be found at https://github.com/vlarsson/homopt
template <typename LossFunction>
class HomographyJacobianAccumulator {
  public:
    HomographyJacobianAccumulator(
        const cv::Mat& correspondences_,
        const size_t* sample_,
        const size_t& sample_size_,
        const LossFunction &l,
        const double *w = nullptr) : 
            correspondences(&correspondences_), 
            sample(sample_),
            sample_size(sample_size_),
            loss_fn(l),
            weights(w) {}

    double residual(const Eigen::Matrix3d &H) const 
    {
        double cost = 0.0;

        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        Eigen::Vector2d pt1, pt2;
        for (size_t k = 0; k < sample_size; ++k) 
        {
            const size_t& point_idx = 
                sample == nullptr ? k : sample[k];

            const double &x1_0 = correspondences->at<double>(point_idx, 0), 
                &x1_1 = correspondences->at<double>(point_idx, 1);
            const double x2_0 = correspondences->at<double>(point_idx, 2), 
                &x2_1 =correspondences->at<double>(point_idx, 3);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double r0 = Hx1_0 * inv_Hx1_2 - x2_0;
            const double r1 = Hx1_1 * inv_Hx1_2 - x2_1;
            const double r2 = r0 * r0 + r1 * r1;

            if (weights == nullptr)
                cost += loss_fn.loss(r2);
            else
                cost += weights[k] * loss_fn.loss(r2);
        }
        return cost;
    }

    void accumulate(const Eigen::Matrix3d &H, Eigen::Matrix<double, 8, 8> &JtJ, Eigen::Matrix<double, 8, 1> &Jtr) const {
        Eigen::Matrix<double, 2, 8> dH;
        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        for (size_t k = 0; k < sample_size; ++k) 
        {
            const size_t& point_idx = 
                sample == nullptr ? k : sample[k];

            const double &x1_0 = correspondences->at<double>(point_idx, 0), 
                &x1_1 = correspondences->at<double>(point_idx, 1);
            const double x2_0 = correspondences->at<double>(point_idx, 2), 
                &x2_1 =correspondences->at<double>(point_idx, 3);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double z0 = Hx1_0 * inv_Hx1_2;
            const double z1 = Hx1_1 * inv_Hx1_2;

            const double r0 = z0 - x2_0;
            const double r1 = z1 - x2_1;
            const double r2 = r0 * r0 + r1 * r1;

            // Compute weight from robust loss function (used in the IRLS)
            double weight = loss_fn.weight(r2) / sample_size;
            if (weights != nullptr)
                weight = weights[k] * weight;

            if(weight == 0.0)
                continue;

            dH << x1_0, 0.0, -x1_0 * z0, x1_1, 0.0, -x1_1 * z0, 1.0, 0.0, // -z0,
                0.0, x1_0, -x1_0 * z1, 0.0, x1_1, -x1_1 * z1, 0.0, 1.0;   // -z1,
            dH = dH * inv_Hx1_2;

            // accumulate into JtJ and Jtr
            Jtr += dH.transpose() * (weight * Eigen::Vector2d(r0, r1));
            for (size_t i = 0; i < 8; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * dH.col(i).dot(dH.col(j));
                }
            }
        }
    }

    Eigen::Matrix3d step(Eigen::Matrix<double, 8, 1> dp, const Eigen::Matrix3d &H) const {
        Eigen::Matrix3d H_new = H;
        Eigen::Map<Eigen::Matrix<double, 8, 1>>(H_new.data()) += dp;
        return H_new;
    }
    typedef Eigen::Matrix3d param_t;
    static constexpr size_t num_params = 8;

    private:
        const cv::Mat* correspondences;
        const size_t* sample;
        const size_t sample_size;

        const LossFunction &loss_fn;
        const double *weights;
};

template <typename LossFunction>
class RelativePoseJacobianAccumulator {
  public:
    RelativePoseJacobianAccumulator(
        const cv::Mat& correspondences_,
        const size_t* sample_,
        const size_t& sample_size_,
        const LossFunction &l,
        const double *w = nullptr) : 
            correspondences(&correspondences_), 
            sample(sample_),
            sample_size(sample_size_),
            loss_fn(l),
            weights(w) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        Eigen::Vector2d pt1, pt2;
        double cost = 0.0;
        for (size_t k = 0; k < sample_size; ++k) 
        {
            const size_t point_idx = sample == nullptr ? k : sample[k];

            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);

            double C = pt2.homogeneous().dot(E * pt1.homogeneous());
            double nJc_sq = (E.block<2, 3>(0, 0) * pt1.homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * pt2.homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            if (weights == nullptr)
                cost += loss_fn.loss(r2);
            else
                cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    void accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ, 
        Eigen::Matrix<double, 5, 1> &Jtr, 
        Eigen::Matrix<double, 3, 2> &tangent_basis) const {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(pose.t.x()) < std::abs(pose.t.y())) {
            // x < y
            if (std::abs(pose.t.x()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose.t.y()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose.t).normalized();

        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R)
        Eigen::Matrix3d R = pose.R();
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        for (size_t k = 0; k < sample_size; ++k) 
        {
            const size_t& point_idx = sample == nullptr ? k : sample[k];

            Eigen::Vector2d pt1, pt2;
            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);

            double C = pt2.homogeneous().dot(E * pt1.homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * pt2.homogeneous(), E.block<2, 3>(0, 0) * pt1.homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            double weight = loss_fn.weight(r * r) / sample_size;
            if (weights != nullptr)
                weight = weights[k] * weight;

            if(weight == 0.0)
                continue;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << pt1(0) * pt2(0), pt1(0) * pt2(1), pt1(0), pt1(1) * pt2(0), pt1(1) * pt2(1), pt1(1), pt2(0), pt2(1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * pt1(0) + J_C(0) * pt2(0));
            dF(1) -= s * (J_C(3) * pt1(0) + J_C(0) * pt2(1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * pt1(1) + J_C(1) * pt2(0));
            dF(4) -= s * (J_C(3) * pt1(1) + J_C(1) * pt2(1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 5> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
        }
    }

    private:
        const cv::Mat* correspondences;
        const size_t* sample;
        const size_t sample_size;

        const LossFunction &loss_fn;
        const double *weights;
};


template <typename LossFunction>
class SingleFocalRelativePoseJacobianAccumulator {
  public:
    SingleFocalRelativePoseJacobianAccumulator(
        const cv::Mat& correspondences_,
        const size_t* sample_,
        const size_t& sample_size_,
        const LossFunction &l,
        const double *w = nullptr) : 
            correspondences(&correspondences_), 
            sample(sample_),
            sample_size(sample_size_),
            loss_fn(l),
            weights(w) {}

    double residual(const CameraPose &pose) const 
    {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);
        Eigen::Matrix3d K_inv;
        K_inv << 1.0 / pose.focal_length, 0.0, 0.0, 
            0.0, 1.0 / pose.focal_length, 0.0, 
            0.0, 0.0, 1.0;
        Eigen::Matrix3d F = K_inv * (E * K_inv);

        Eigen::Vector2d pt1, pt2;
        double cost = 0.0;
        for (size_t k = 0; k < sample_size; ++k) 
        {
            const size_t point_idx = sample == nullptr ? k : sample[k];
            
            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);

            double C = pt2.homogeneous().dot(F * pt1.homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * pt1.homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * pt2.homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            if (weights == nullptr)
                cost += loss_fn.loss(r2);
            else
                cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const CameraPose &pose, 
        Eigen::Matrix<double, 6, 6> &JtJ, 
        Eigen::Matrix<double, 6, 1> &Jtr) 
    {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(pose.t.x()) < std::abs(pose.t.y())) {
            // x < y
            if (std::abs(pose.t.x()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose.t.y()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose.t).normalized();

        double f_inv = 1.0 / pose.focal_length;
        double f_inv_sq = f_inv * f_inv;
        Eigen::Matrix3d K_inv;
        K_inv << f_inv, 0.0, 0.0, 0.0, f_inv, 0.0, 0.0, 0.0, 1;
        
        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);
        Eigen::Matrix3d F = K_inv * (E * K_inv);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);        
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        dR.row(0) *= f_inv_sq;
        dR.row(1) *= f_inv_sq;
        dR.row(2) *= f_inv;
        dR.row(3) *= f_inv_sq;
        dR.row(4) *= f_inv_sq;
        dR.row(5) *= f_inv;
        dR.row(6) *= f_inv;
        dR.row(7) *= f_inv;


        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        dt.row(0) *= f_inv_sq;
        dt.row(1) *= f_inv_sq;
        dt.row(2) *= f_inv;
        dt.row(3) *= f_inv_sq;
        dt.row(4) *= f_inv_sq;
        dt.row(5) *= f_inv;
        dt.row(6) *= f_inv;
        dt.row(7) *= f_inv;

        Eigen::Matrix<double, 9, 1> df;

        double f_inv_cubed = f_inv * f_inv * f_inv;

        //df << -F(0, 0) * 2.0 * f_inv_cubed, -F(0, 1) * 2.0 * f_inv_cubed, -F(0, 2) * f_inv_sq, 
        //    -F(1, 0) * 2.0 * f_inv_cubed, -F(1, 1) * 2.0 * f_inv_cubed, -F(1, 2) * f_inv_sq,
        //    -F(2, 0) * f_inv_sq, -F(2, 1) * f_inv_sq, 0;

         //df << -E(0, 0) * 2.0 * f_inv_cubed, -E(0, 1) * 2.0 * f_inv_cubed, -E(0, 2) * f_inv_sq, 
         //   -E(1, 0) * 2.0 * f_inv_cubed, -E(1, 1) * 2.0 * f_inv_cubed, -E(1, 2) * f_inv_sq,
         //   -E(2, 0) * f_inv_sq, -E(2, 1) * f_inv_sq, 0;

        // df << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        df << -E(0, 0) * 2.0 * f_inv_cubed, -E(1, 0) * 2.0 * f_inv_cubed, -E(2, 0) * f_inv_sq, 
              -E(0, 1) * 2.0 * f_inv_cubed, -E(1, 1) * 2.0 * f_inv_cubed, -E(2, 1) * f_inv_sq,
              -E(0, 2) * f_inv_sq, -E(1, 2) * f_inv_sq, 0;

        size_t num_residuals = 0;
        
        for (size_t k = 0; k < sample_size; ++k) 
        {
            size_t point_idx;
            if (sample == nullptr)
                point_idx = k;
            else
                point_idx = sample[k];

            Eigen::Vector2d pt1, pt2;
            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);

            double C = pt2.homogeneous().dot(F * pt1.homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * pt2.homogeneous(), F.block<2, 3>(0, 0) * pt1.homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            double weight = loss_fn.weight(r * r);
            if (weights != nullptr)
                weight = weights[k] * weight;

            if (weight == 0.0) 
                continue;
            
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << pt1(0) * pt2(0), pt1(0) * pt2(1), pt1(0), pt1(1) * pt2(0), pt1(1) * pt2(1),
                pt1(1), pt2(0), pt2(1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * pt1(0) + J_C(0) * pt2(0));
            dF(1) -= s * (J_C(3) * pt1(0) + J_C(0) * pt2(1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * pt1(1) + J_C(1) * pt2(0));
            dF(4) -= s * (J_C(3) * pt1(1) + J_C(1) * pt2(1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;
            J(0, 5) = dF * df;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
            JtJ(5, 0) += weight * (J(5) * J(0));
            JtJ(5, 1) += weight * (J(5) * J(1));
            JtJ(5, 2) += weight * (J(5) * J(2));
            JtJ(5, 3) += weight * (J(5) * J(3));
            JtJ(5, 4) += weight * (J(5) * J(4));
            JtJ(5, 5) += weight * (J(5) * J(5));
        }
        return num_residuals;
    }

    inline Eigen::Vector4d quat_exp(const Eigen::Vector3d &w) const {
        const double theta2 = w.squaredNorm();
        const double theta = std::sqrt(theta2);
        const double theta_half = 0.5 * theta;

        double re, im;
        if (theta > 1e-6) {
            re = std::cos(theta_half);
            im = std::sin(theta_half) / theta;
        } else {
            // we are close to zero, use taylor expansion to avoid problems
            // with zero divisors in sin(theta/2)/theta
            const double theta4 = theta2 * theta2;
            re = 1.0 - (1.0 / 8.0) * theta2 + (1.0 / 384.0) * theta4;
            im = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4;

            // for the linearized part we re-normalize to ensure unit length
            // here s should be roughly 1.0 anyways, so no problem with zero div
            const double s = std::sqrt(re * re + im * im * theta2);
            re /= s;
            im /= s;
        }
        return Eigen::Vector4d(re, im * w(0), im * w(1), im * w(2));
    }

    inline Eigen::Vector4d quat_step_post(const Eigen::Vector4d &q, const Eigen::Vector3d &w_delta) const {
        return quat_multiply(q, quat_exp(w_delta));
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const 
    {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        pose_new.focal_length = std::max(pose.focal_length + dp(5, 0), 0.0);        
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const cv::Mat* correspondences;
    const size_t* sample;
    const size_t sample_size;

    const LossFunction &loss_fn;
    const double *weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

// This is the SVD factorization proposed by Bartoli and Sturm in
// Non-Linear Estimation of the Fundamental Matrix With Minimal Parameters, PAMI 2004
// Though we do different updates (lie vs the euler angles used in the original paper)
struct FactorizedFundamentalMatrix {
    FactorizedFundamentalMatrix() {}
    FactorizedFundamentalMatrix(const Eigen::Matrix3d &F) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullV | Eigen::ComputeFullU);
        U = svd.matrixU();
        V = svd.matrixV();
        Eigen::Vector3d s = svd.singularValues();
        sigma = s(1) / s(0);
    }
    Eigen::Matrix3d F() const {
        return U.col(0) * V.col(0).transpose() + sigma * U.col(1) * V.col(1).transpose();
    }

    Eigen::Matrix3d U, V;
    double sigma;
};

template <typename LossFunction>
class FundamentalJacobianAccumulator {
  public:
    FundamentalJacobianAccumulator(
        const cv::Mat& correspondences_,
        const size_t* sample_,
        const size_t& sample_size_,
        const LossFunction &l,
        const double *w = nullptr)  : 
            correspondences(&correspondences_), 
            sample(sample_),
            sample_size(sample_size_),
            loss_fn(l),
            weights(w) {}

    double residual(const FactorizedFundamentalMatrix &FF) const {
        Eigen::Matrix3d F = FF.F();

        Eigen::Vector2d pt1, pt2;
        double cost = 0.0;
        for (size_t k = 0; k < sample_size; ++k) 
        {
            size_t point_idx;
            if (sample == nullptr)
                point_idx = k;
            else
                point_idx = sample[k];

            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);
            
            double C = pt2.homogeneous().dot(F * pt1.homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * pt1.homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * pt2.homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            if (weights == nullptr)
                cost += loss_fn.loss(r2);
            else
                cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    void accumulate(const FactorizedFundamentalMatrix &FF, Eigen::Matrix<double, 7, 7> &JtJ, Eigen::Matrix<double, 7, 1> &Jtr) const {

        Eigen::Matrix3d F = FF.F();

        // Matrices contain the jacobians of F w.r.t. the factorized fundamental matrix (U,V,sigma)
        Eigen::Matrix3d d_sigma = FF.U.col(1) * FF.V.col(1).transpose();
        Eigen::Matrix<double, 9, 7> dF_dparams;
        dF_dparams << 0, F(2, 0), -F(1, 0), 0, F(0, 2), -F(0, 1), d_sigma(0, 0),
            -F(2, 0), 0, F(0, 0), 0, F(1, 2), -F(1, 1), d_sigma(1, 0),
            F(1, 0), -F(0, 0), 0, 0, F(2, 2), -F(2, 1), d_sigma(2, 0),
            0, F(2, 1), -F(1, 1), -F(0, 2), 0, F(0, 0), d_sigma(0, 1),
            -F(2, 1), 0, F(0, 1), -F(1, 2), 0, F(1, 0), d_sigma(1, 1),
            F(1, 1), -F(0, 1), 0, -F(2, 2), 0, F(2, 0), d_sigma(2, 1),
            0, F(2, 2), -F(1, 2), F(0, 1), -F(0, 0), 0, d_sigma(0, 2),
            -F(2, 2), 0, F(0, 2), F(1, 1), -F(1, 0), 0, d_sigma(1, 2),
            F(1, 2), -F(0, 2), 0, F(2, 1), -F(2, 0), 0, d_sigma(2, 2);

        for (size_t k = 0; k < sample_size; ++k) 
        {
            size_t point_idx;
            if (sample == nullptr)
                point_idx = k;
            else
                point_idx = sample[k];

            Eigen::Vector2d pt1, pt2;
            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);

            double C = pt2.homogeneous().dot(F * pt1.homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * pt2.homogeneous(), F.block<2, 3>(0, 0) * pt1.homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            double weight = loss_fn.weight(r * r) / sample_size;
            
            // Multiplying by the provided weights if they are available 
            if (weights != nullptr)
                weight = weights[k] * weight;
            if (weight == 0.0)
                continue;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << pt1(0) * pt2(0), pt1(0) * pt2(1), pt1(0), pt1(1) * pt2(0), pt1(1) * pt2(1), pt1(1), pt2(0), pt2(1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * pt1(0) + J_C(0) * pt2(0));
            dF(1) -= s * (J_C(3) * pt1(0) + J_C(0) * pt2(1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * pt1(1) + J_C(1) * pt2(0));
            dF(4) -= s * (J_C(3) * pt1(1) + J_C(1) * pt2(1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 7> J = dF * dF_dparams;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            for (size_t i = 0; i < 7; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J(i) * J(j));
                }
            }
        }
    }

  private:
    const cv::Mat* correspondences;
    const size_t* sample;
    const size_t sample_size;

    const LossFunction &loss_fn;
    const double *weights;
};

} // namespace pose_lib

#endif