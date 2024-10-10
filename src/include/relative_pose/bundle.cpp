#include "bundle.h"
#include "jacobian_impl.h"
#include "robust_loss.h"
#include "colmap_models.h"
#include <opencv2/core.hpp>

#include <iostream>

namespace pose_lib {

#define SWITCH_LOSS_FUNCTIONS                     \
    case BundleOptions::LossType::TRIVIAL:        \
        SWITCH_LOSS_FUNCTION_CASE(TrivialLoss);   \
        break;                                    \
    case BundleOptions::LossType::TRUNCATED:      \
        SWITCH_LOSS_FUNCTION_CASE(TruncatedLoss); \
        break;                                    \
    case BundleOptions::LossType::HUBER:          \
        SWITCH_LOSS_FUNCTION_CASE(HuberLoss);     \
        break;                                    \
    case BundleOptions::LossType::CAUCHY:         \
        SWITCH_LOSS_FUNCTION_CASE(CauchyLoss);    \
        break;

typedef std::function<void(const BundleStats &stats)> IterationCallback;

// Callback which prints debug info from the iterations
void print_iteration(const BundleStats &stats) {
    if (stats.iterations == 0) {
        std::cout << "initial_cost=" << stats.initial_cost << "\n";
    }
    std::cout << "iter=" << stats.iterations << ", cost=" << stats.cost << ", step=" << stats.step_norm
              << ", grad=" << stats.grad_norm << ", lambda=" << stats.lambda << "\n";
}

template <typename LossFunction> IterationCallback setup_callback(const BundleOptions &opt, LossFunction &loss_fn) {
    if (opt.verbose) {
        return print_iteration;
    } else {
        return nullptr;
    }
}

template <typename JacobianAccumulator>
int lm_pnp_impl(const JacobianAccumulator &accum, 
    CameraPose *pose, 
    const BundleOptions &opt) 
{
    Eigen::Matrix<double, 6, 6> JtJ;
    Eigen::Matrix<double, 6, 1> Jtr;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw;
    sw.setZero();

    // Compute initial cost
    double cost = accum.residual(*pose);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            accum.accumulate(*pose, JtJ, Jtr);
            if (Jtr.norm() < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        JtJ(0, 0) += lambda;
        JtJ(1, 1) += lambda;
        JtJ(2, 2) += lambda;
        JtJ(3, 3) += lambda;
        JtJ(4, 4) += lambda;
        JtJ(5, 5) += lambda;

        Eigen::Matrix<double, 6, 1> sol = -JtJ.selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        if (sol.norm() < opt.step_tol) {
            break;
        }

        Eigen::Vector3d w = sol.block<3, 1>(0, 0);
        const double theta = w.norm();
        w /= theta;
        const double a = std::sin(theta);
        const double b = std::cos(theta);
        sw(0, 1) = -w(2);
        sw(0, 2) = w(1);
        sw(1, 2) = -w(0);
        sw(1, 0) = w(2);
        sw(2, 0) = -w(1);
        sw(2, 1) = w(0);

        CameraPose pose_new;
        Eigen::Matrix3d R = pose->R();
        pose_new.R = R + R * (a * sw + (1 - b) * sw * sw);
        pose_new.t = pose->t + R * sol.block<3, 1>(3, 0);
        double cost_new = accum.residual(pose_new);

        if (cost_new < cost) {
            *pose = pose_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            JtJ(0, 0) -= lambda;
            JtJ(1, 1) -= lambda;
            JtJ(2, 2) -= lambda;
            JtJ(3, 3) -= lambda;
            JtJ(4, 4) -= lambda;
            JtJ(5, 5) -= lambda;
            lambda *= 10;
            recompute_jac = false;
        }
    }

    return iter;
}

template <typename Problem, typename Param = typename Problem::param_t>
BundleStats lm_impl(Problem &problem, Param *parameters, const BundleOptions &opt,
                    IterationCallback callback = nullptr) {
    constexpr int n_params = Problem::num_params;
    Eigen::Matrix<double, n_params, n_params> JtJ;
    Eigen::Matrix<double, n_params, 1> Jtr;

    // Initialize
    BundleStats stats;
    stats.cost = problem.residual(*parameters);
    stats.initial_cost = stats.cost;
    stats.grad_norm = -1;
    stats.step_norm = -1;
    stats.invalid_steps = 0;
    stats.lambda = opt.initial_lambda;

    bool recompute_jac = true;
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; ++stats.iterations) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            problem.accumulate(*parameters, JtJ, Jtr);
            stats.grad_norm = Jtr.norm();
            if (stats.grad_norm < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        for (size_t k = 0; k < n_params; ++k) {
            JtJ(k, k) += stats.lambda;
        }

        Eigen::Matrix<double, n_params, 1> sol = -JtJ.template selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        stats.step_norm = sol.norm();
        if (stats.step_norm < opt.step_tol) {
            break;
        }

        Param parameters_new = problem.step(sol, *parameters);

        double cost_new = problem.residual(parameters_new);

        if (cost_new < stats.cost) {
            *parameters = parameters_new;
            stats.lambda = std::max(opt.min_lambda, stats.lambda / 10);
            stats.cost = cost_new;
            recompute_jac = true;
        } else {
            stats.invalid_steps++;
            // Remove dampening
            for (size_t k = 0; k < n_params; ++k) {
                JtJ(k, k) -= stats.lambda;
            }
            stats.lambda = std::min(opt.max_lambda, stats.lambda * 10);
            recompute_jac = false;
        }
        if (callback != nullptr) {
            callback(stats);
        }
    }
    return stats;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// 3 View relative pose refinement

template <typename WeightType, typename LossFunction>
BundleStats refine_3v_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                              const std::vector<Point2D> &x3, ThreeViewCameraPose *pose,
                              const BundleOptions &opt, const WeightType &weights) {
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    ThreeViewRelativePoseJacobianAccumulator<LossFunction, WeightType> accum(x1, x2, x3, loss_fn, weights);
    return lm_impl<decltype(accum)>(accum, pose, opt, callback);
}

template <typename WeightType>
BundleStats refine_3v_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                              const std::vector<Point2D> &x3, ThreeViewCameraPose *pose,
                              const BundleOptions &opt, const WeightType &weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return refine_3v_relpose<WeightType, LossFunction>(x1, x2, x3, pose, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for essential matrix refinement
BundleStats refine_3v_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                              const std::vector<Point2D> &x3, ThreeViewCameraPose *pose,
                              const BundleOptions &opt, const std::vector<double> &weights) {
    if (weights.size() == x1.size()) {
        return refine_3v_relpose<std::vector<double>>(x1, x2, x3, pose, opt, weights);
    } else {
        return refine_3v_relpose<UniformWeightVector>(x1, x2, x3, pose, opt, UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////



template <typename JacobianAccumulator>
int lm_6dof_impl(const JacobianAccumulator &accum, CameraPose *pose, const BundleOptions &opt) {
    Eigen::Matrix<double, 6, 6> JtJ;
    Eigen::Matrix<double, 6, 1> Jtr;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw;
    sw.setZero();

    // Compute initial cost
    double cost = accum.residual(*pose);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            accum.accumulate(*pose, JtJ, Jtr);
            if (Jtr.norm() < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        JtJ(0, 0) += lambda;
        JtJ(1, 1) += lambda;
        JtJ(2, 2) += lambda;
        JtJ(3, 3) += lambda;
        JtJ(4, 4) += lambda;
        JtJ(5, 5) += lambda;

        Eigen::Matrix<double, 6, 1> sol = -JtJ.selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        if (sol.norm() < opt.step_tol) {
            break;
        }

        Eigen::Vector3d w = sol.block<3, 1>(0, 0);
        const double theta = w.norm();
        w /= theta;
        const double a = std::sin(theta);
        const double b = std::cos(theta);
        sw(0, 1) = -w(2);
        sw(0, 2) = w(1);
        sw(1, 2) = -w(0);
        sw(1, 0) = w(2);
        sw(2, 0) = -w(1);
        sw(2, 1) = w(0);

        CameraPose pose_new;
        Eigen::Matrix3d R = pose->R();
        pose_new.R = R + R * (a * sw + (1 - b) * sw * sw);
        pose_new.t = pose->t + R * sol.block<3, 1>(3, 0);
        double cost_new = accum.residual(pose_new);

        if (cost_new < cost) {
            *pose = pose_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            JtJ(0, 0) -= lambda;
            JtJ(1, 1) -= lambda;
            JtJ(2, 2) -= lambda;
            JtJ(3, 3) -= lambda;
            JtJ(4, 4) -= lambda;
            JtJ(5, 5) -= lambda;
            lambda *= 10;
            recompute_jac = false;
        }
    }

    return iter;
}

template <typename JacobianAccumulator>
int lm_5dof_f_impl(JacobianAccumulator &accum, CameraPose *pose, const BundleOptions &opt) 
{
    Eigen::Matrix<double, 6, 6> JtJ;
    Eigen::Matrix<double, 6, 1> Jtr;
    Eigen::Matrix<double, 3, 2> tangent_basis;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw;
    sw.setZero();

    // Compute initial cost
    double cost = accum.residual(*pose);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) 
    {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            accum.accumulate(*pose, JtJ, Jtr);
            if (Jtr.norm() < opt.gradient_tol) 
                break;
        }

        // Add dampening
        JtJ(0, 0) += lambda;
        JtJ(1, 1) += lambda;
        JtJ(2, 2) += lambda;
        JtJ(3, 3) += lambda;
        JtJ(4, 4) += lambda;

        Eigen::Matrix<double, 6, 1> sol = -JtJ.selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        if (sol.norm() < opt.step_tol) {
            break;
        }

        Eigen::Vector3d w = sol.block<3, 1>(0, 0);
        const double theta = w.norm();
        w /= theta;
        const double a = std::sin(theta);
        const double b = std::cos(theta);
        sw(0, 1) = -w(2);
        sw(0, 2) = w(1);
        sw(1, 2) = -w(0);
        sw(1, 0) = w(2);
        sw(2, 0) = -w(1);
        sw(2, 1) = w(0);

        CameraPose pose_new;
        Eigen::Matrix3d R = pose->R();
        pose_new.q = pose_new.rotmat_to_quat(R + R * (a * sw + (1 - b) * sw * sw));
        // In contrast to the 6dof case, we don't apply R here
        // (since this can already be added into tangent_basis)
        pose_new.t = pose->t + tangent_basis * sol.block<2, 1>(3, 0);
        double cost_new = accum.residual(pose_new);

        if (cost_new < cost) {
            *pose = pose_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            JtJ(0, 0) -= lambda;
            JtJ(1, 1) -= lambda;
            JtJ(2, 2) -= lambda;
            JtJ(3, 3) -= lambda;
            JtJ(4, 4) -= lambda;
            lambda *= 10;
            recompute_jac = false;
        }
    }

    return iter;
}

template <typename JacobianAccumulator>
int lm_5dof_impl(const JacobianAccumulator &accum, CameraPose *pose, const BundleOptions &opt) {
    Eigen::Matrix<double, 5, 5> JtJ;
    Eigen::Matrix<double, 5, 1> Jtr;
    Eigen::Matrix<double, 3, 2> tangent_basis;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw;
    sw.setZero();

    // Compute initial cost
    double cost = accum.residual(*pose);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            accum.accumulate(*pose, JtJ, Jtr, tangent_basis);
            if (Jtr.norm() < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        JtJ(0, 0) += lambda;
        JtJ(1, 1) += lambda;
        JtJ(2, 2) += lambda;
        JtJ(3, 3) += lambda;
        JtJ(4, 4) += lambda;

        Eigen::Matrix<double, 5, 1> sol = -JtJ.selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        if (sol.norm() < opt.step_tol) {
            break;
        }

        Eigen::Vector3d w = sol.block<3, 1>(0, 0);
        const double theta = w.norm();
        w /= theta;
        const double a = std::sin(theta);
        const double b = std::cos(theta);
        sw(0, 1) = -w(2);
        sw(0, 2) = w(1);
        sw(1, 2) = -w(0);
        sw(1, 0) = w(2);
        sw(2, 0) = -w(1);
        sw(2, 1) = w(0);

        CameraPose pose_new;
        Eigen::Matrix3d R = pose->R();
        pose_new.q = pose_new.rotmat_to_quat(R + R * (a * sw + (1 - b) * sw * sw));
        // In contrast to the 6dof case, we don't apply R here
        // (since this can already be added into tangent_basis)
        pose_new.t = pose->t + tangent_basis * sol.block<2, 1>(3, 0);
        double cost_new = accum.residual(pose_new);

        if (cost_new < cost) {
            *pose = pose_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            JtJ(0, 0) -= lambda;
            JtJ(1, 1) -= lambda;
            JtJ(2, 2) -= lambda;
            JtJ(3, 3) -= lambda;
            JtJ(4, 4) -= lambda;
            lambda *= 10;
            recompute_jac = false;
        }
    }

    return iter;
}

template <typename JacobianAccumulator>
int lm_F_impl(const JacobianAccumulator &accum, Eigen::Matrix3d *fundamental_matrix, const BundleOptions &opt) 
{
    Eigen::Matrix<double, 7, 7> JtJ;
    Eigen::Matrix<double, 7, 1> Jtr;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw1, sw2;
    sw1.setZero();
    sw2.setZero();

    // compute factorization which is used for the optimization
    FactorizedFundamentalMatrix F(*fundamental_matrix);

    // Compute initial cost
    double cost = accum.residual(F);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            accum.accumulate(F, JtJ, Jtr);
            if (Jtr.norm() < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        for (size_t k = 0; k < 7; ++k) {
            JtJ(k, k) += lambda;
        }

        Eigen::Matrix<double, 7, 1> sol = -JtJ.selfadjointView<Eigen::Lower>().llt().solve(Jtr);
        if (sol.norm() < opt.step_tol) {
            break;
        }

        // Update U and V
        Eigen::Vector3d w1 = sol.block<3, 1>(0, 0);
        Eigen::Vector3d w2 = sol.block<3, 1>(3, 0);
        const double theta1 = w1.norm();
        const double theta2 = w2.norm();
        w1 /= theta1;
        w2 /= theta2;
        const double a1 = std::sin(theta1);
        const double b1 = std::cos(theta1);
        sw1(0, 1) = -w1(2);
        sw1(0, 2) = w1(1);
        sw1(1, 2) = -w1(0);
        sw1(1, 0) = w1(2);
        sw1(2, 0) = -w1(1);
        sw1(2, 1) = w1(0);
        const double a2 = std::sin(theta2);
        const double b2 = std::cos(theta2);
        sw2(0, 1) = -w2(2);
        sw2(0, 2) = w2(1);
        sw2(1, 2) = -w2(0);
        sw2(1, 0) = w2(2);
        sw2(2, 0) = -w2(1);
        sw2(2, 1) = w2(0);

        FactorizedFundamentalMatrix F_new;
        F_new.U = F.U + (a1 * sw1 + (1 - b1) * sw1 * sw1) * F.U;
        F_new.V = F.V + (a2 * sw2 + (1 - b2) * sw2 * sw2) * F.V;
        F_new.sigma = F.sigma + sol(6);

        double cost_new = accum.residual(F_new);

        if (cost_new < cost) {
            F = F_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            // Remove dampening
            for (size_t k = 0; k < 7; ++k) {
                JtJ(k, k) -= lambda;
            }
            lambda *= 10;
            recompute_jac = false;
        }
    }

    *fundamental_matrix = F.F();

    return iter;
}

template <typename JacobianAccumulator>
int lm_H_impl(const JacobianAccumulator &accum, Eigen::Matrix3d *homography, const BundleOptions &opt) 
{
    Eigen::Matrix<double, 8, 8> JtJ;
    Eigen::Matrix<double, 8, 1> Jtr;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw1, sw2;
    sw1.setZero();
    sw2.setZero();

    // compute factorization which is used for the optimization
    Eigen::Matrix3d H = *homography;

    // Compute initial cost
    double cost = accum.residual(H);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            accum.accumulate(H, JtJ, Jtr);
            if (Jtr.norm() < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        for (size_t k = 0; k < 8; ++k) {
            JtJ(k, k) += lambda;
        }

        Eigen::Matrix<double, 8, 1> sol = -JtJ.selfadjointView<Eigen::Lower>().llt().solve(Jtr);
        if (sol.norm() < opt.step_tol) {
            break;
        }

        Eigen::Matrix3d H_new;
        H_new(0, 0) = sol(0);
        H_new(0, 1) = sol(1);
        H_new(0, 2) = sol(2);
        H_new(1, 0) = sol(3);
        H_new(1, 1) = sol(4);
        H_new(1, 2) = sol(5);
        H_new(2, 0) = sol(6);
        H_new(2, 1) = sol(7);
        H_new(2, 2) = 1;

        double cost_new = accum.residual(H_new);

        if (cost_new < cost) {
            H = H_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            // Remove dampening
            for (size_t k = 0; k < 8; ++k) {
                JtJ(k, k) -= lambda;
            }
            lambda *= 10;
            recompute_jac = false;
        }
    }

    *homography = H;

    return iter;
}

int refine_relpose_single_focal(const cv::Mat &correspondences_,
                    const size_t *sample_,
                    const size_t &sample_size_,
                    CameraPose *pose, 
                   double &focal_length,
                    const BundleOptions &opt,
                    const double* weights)
{
    if (weights != nullptr) 
    {
        // We have per-residual weights
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                             \
        {                                                                                                       \
            LossFunction loss_fn(opt.loss_scale);                                                               \
            SingleFocalRelativePoseJacobianAccumulator<LossFunction> accum(correspondences_, sample_, sample_size_, loss_fn, weights); \
            return lm_5dof_f_impl<decltype(accum)>(accum, pose, opt);                                             \
        }
            SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE
        default:
            return -1;
        };
    } else {
        // Uniformly weighted residuals
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                               \
        {                                                                         \
            LossFunction loss_fn(opt.loss_scale);                                 \
            SingleFocalRelativePoseJacobianAccumulator<LossFunction> accum(correspondences_, sample_, sample_size_, loss_fn); \
            return lm_5dof_f_impl<decltype(accum)>(accum, pose, opt);               \
        }
            SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}

int refine_relpose(const cv::Mat &correspondences_,
                    const size_t *sample_,
                    const size_t &sample_size_,
                    CameraPose *pose, 
                    const BundleOptions &opt,
                    const double* weights)
{
    if (weights != nullptr) 
    {
        // We have per-residual weights
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                             \
        {                                                                                                       \
            LossFunction loss_fn(opt.loss_scale);                                                               \
            RelativePoseJacobianAccumulator<LossFunction> accum(correspondences_, sample_, sample_size_, loss_fn, weights); \
            return lm_5dof_impl<decltype(accum)>(accum, pose, opt);                                             \
        }
            SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE
        default:
            return -1;
        };
    } else {
        // Uniformly weighted residuals
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                               \
        {                                                                         \
            LossFunction loss_fn(opt.loss_scale);                                 \
            RelativePoseJacobianAccumulator<LossFunction> accum(correspondences_, sample_, sample_size_, loss_fn); \
            return lm_5dof_impl<decltype(accum)>(accum, pose, opt);               \
        }
            SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}

int refine_fundamental(const cv::Mat &correspondences_,
                       const size_t *sample_,
                       const size_t &sample_size_,
                       Eigen::Matrix3d *pose,
                       const BundleOptions &opt,
                       const double *weights) {

    if (weights != nullptr) 
    {
        // We have per-residual weights

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                            \
    {                                                                                                      \
        LossFunction loss_fn(opt.loss_scale);                                                              \
        FundamentalJacobianAccumulator<LossFunction> accum(correspondences_, sample_, sample_size_, loss_fn, weights); \
        return lm_F_impl<decltype(accum)>(accum, pose, opt);                                               \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    } else {

        // Uniformly weighted residuals
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                              \
        {                                                                        \
            LossFunction loss_fn(opt.loss_scale);                                \
            FundamentalJacobianAccumulator<LossFunction> accum(correspondences_, sample_, sample_size_, loss_fn); \
            return lm_F_impl<decltype(accum)>(accum, pose, opt);                 \
        }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}

int refine_homography(    
    const cv::Mat &correspondences_,
    const size_t *sample_,
    const size_t &sample_size_, 
    Eigen::Matrix3d *H,
    const BundleOptions &opt,
    const double *weights)
{
    if (weights != nullptr) 
    {
        // We have per-residual weights

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                            \
    {                                                                                                      \
        LossFunction loss_fn(opt.loss_scale);                                                              \
        HomographyJacobianAccumulator<LossFunction> accum(correspondences_, sample_, sample_size_, loss_fn, weights); \
        return lm_H_impl<decltype(accum)>(accum, H, opt);                                               \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    } else {

        // Uniformly weighted residuals
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                              \
        {                                                                        \
            LossFunction loss_fn(opt.loss_scale);                                \
            HomographyJacobianAccumulator<LossFunction> accum(correspondences_, sample_, sample_size_, loss_fn); \
            return lm_H_impl<decltype(accum)>(accum, H, opt);                 \
        }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////
// Absolute pose with points (PnP)
// Interface for calibrated camera
int refine_pnp(
    const cv::Mat &correspondences_,
    const size_t *sample_,
    const size_t &sample_size_, 
    CameraPose *pose,
    const BundleOptions &opt,
    const double *weights) 
{
    pose_lib::Camera camera;
    camera.model_id = NullCameraModel::model_id;
    return refine_pnp(
            correspondences_, 
            sample_,
            sample_size_,
            camera, 
            pose, 
            opt, 
            weights);
}

template <typename CameraModel, typename LossFunction>
int refine_pnp(
    const cv::Mat &correspondences_,
    const size_t *sample_,
    const size_t &sample_size_, 
    const Camera &camera,
    CameraPose *pose,
    const BundleOptions &opt,
    const double *weights)
{                                      
    LossFunction loss_fn(opt.loss_scale);                                                              
    CameraJacobianAccumulator<CameraModel, LossFunction> accum(
        correspondences_, 
        sample_, 
        sample_size_, 
        loss_fn, 
        weights);

    return lm_pnp_impl<decltype(accum)>(accum, pose, opt);          
}

template <typename CameraModel>
int refine_pnp(
    const cv::Mat &correspondences_,
    const size_t *sample_,
    const size_t &sample_size_, 
    const Camera &camera,
    CameraPose *pose,
    const BundleOptions &opt,
    const double *weights) 
{
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return refine_pnp<CameraModel, LossFunction>(                                                                      \
            correspondences_,                                                                       \
            sample_,                                                                      \
            sample_size_,                                                                      \
            camera,                                                                       \
            pose,                                                                       \
            opt,                                                                       \
            weights);                                                                      \
        SWITCH_LOSS_FUNCTIONS
    default:
        return -1;
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for PnP refinement
int refine_pnp(
    const cv::Mat &correspondences_,
    const size_t *sample_,
    const size_t &sample_size_, 
    const Camera &camera,
    CameraPose *pose,
    const BundleOptions &opt,
    const double *weights) 
{
    switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                             \
        return refine_pnp<Model>(                                                                                  \
            correspondences_,                                                                                   \
            sample_,                                                                                  \
            sample_size_,                                                                                  \
            camera,                                                                                   \
            pose,                                                                                   \
            opt,                                                                                   \
            weights);                                     \
        SWITCH_CAMERA_MODELS
#undef SWITCH_CAMERA_MODEL_CASE
    default:
        return -1;
    }
}
} // namespace pose_lib