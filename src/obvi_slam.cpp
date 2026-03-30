/**
 * @file obvi_bindings.cpp
 * @brief Python bindings for ObVi-SLAM (ROS‑free version) and the adapter implementation.
 *
 * This file contains both the adapter class ObViSlamAdapter and its nanobind
 * wrappers. Combining them into a single translation unit avoids multiple
 * definition errors that arise from including certain ObVi‑SLAM headers
 * in separate .cpp files.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>

#include <refactoring/offline/offline_problem_runner.h>
#include <refactoring/offline/pose_graph_frame_data_adder.h>
#include <refactoring/optimization/object_pose_graph.h>
#include <refactoring/visual_feature_frontend/visual_feature_front_end.h>
#include <run_optimization_utils/run_opt_utils.h>
#include <refactoring/long_term_map/long_term_object_map_extraction.h>
#include <refactoring/long_term_map/long_term_map_factor_creator.h>
#include <refactoring/output_problem_data_extraction.h>
#include <file_io/cv_file_storage/config_file_storage_io.h>
#include <refactoring/bounding_box_frontend/feature_based_bounding_box_front_end.h>
#include <glog/logging.h>
#include <memory>

namespace nb = nanobind;

// Disable automatic type caster for std::pair (needed for RawBoundingBox)
NB_MAKE_OPAQUE(std::pair<vslam_types_refactor::PixelCoord<double>,
    vslam_types_refactor::PixelCoord<double>>);

// -----------------------------------------------------------------------------
// Adapter class
// -----------------------------------------------------------------------------

namespace obvi_adapter {

    /**
     * @brief ROS‑free adapter for ObVi‑SLAM.
     *
     * This class accumulates keyframes (with ORB features) and object detections,
     * and later runs the full offline optimization to refine camera trajectory and
     * object map.
     */
    class ObViSlamAdapter {
    public:
        explicit ObViSlamAdapter(const std::string& config_file);

        void addKeyframe(
            vslam_types_refactor::FrameId frame_id,
            const std::vector<double>& pose_matrix,
            const std::vector<std::pair<double, double>>& keypoints,
            const std::vector<std::vector<unsigned char>>& descriptors,
            const std::unordered_map<vslam_types_refactor::CameraId,
            std::pair<int, int>>&image_sizes);

        void addDetections(
            vslam_types_refactor::FrameId frame_id,
            const std::unordered_map<vslam_types_refactor::CameraId,
            std::vector<vslam_types_refactor::RawBoundingBox>>&detections);

        bool optimize();

        std::unordered_map<vslam_types_refactor::FrameId,
            vslam_types_refactor::Pose3D<double>>
            getOptimizedTrajectory() const;

        std::unordered_map<vslam_types_refactor::ObjectId,
            std::pair<std::string,
            vslam_types_refactor::EllipsoidState<double>>>
            getObjectMap() const;

        void setCameraIntrinsics(
            const std::unordered_map<vslam_types_refactor::CameraId,
            vslam_types_refactor::CameraIntrinsicsMat<double>>&intrinsics);

        void setCameraExtrinsics(
            const std::unordered_map<vslam_types_refactor::CameraId,
            vslam_types_refactor::CameraExtrinsics<double>>&extrinsics);

    private:
        vslam_types_refactor::FullOVSLAMConfig config_;

        std::unordered_map<vslam_types_refactor::CameraId,
            vslam_types_refactor::CameraIntrinsicsMat<double>> intrinsics_;
        std::unordered_map<vslam_types_refactor::CameraId,
            vslam_types_refactor::CameraExtrinsics<double>> extrinsics_;
        std::unordered_map<vslam_types_refactor::FrameId,
            vslam_types_refactor::Pose3D<double>> robot_poses_;
        std::unordered_map<vslam_types_refactor::FrameId,
            std::unordered_map<vslam_types_refactor::CameraId,
            std::vector<vslam_types_refactor::RawBoundingBox>>>
            bounding_boxes_;
        std::unordered_map<vslam_types_refactor::FeatureId,
            vslam_types_refactor::StructuredVisionFeatureTrack>
            visual_features_;
        std::unordered_map<vslam_types_refactor::FrameId,
            std::unordered_map<vslam_types_refactor::CameraId,
            std::pair<int, int>>>
            image_sizes_;
        std::shared_ptr<vslam_types_refactor::MainLtm> long_term_map_;
        vslam_types_refactor::LongTermObjectMapAndResults<vslam_types_refactor::MainLtm> results_;
        vslam_types_refactor::FeatureId next_feature_id_ = 0;
    };

    // -----------------------------------------------------------------------------
    // Implementation
    // -----------------------------------------------------------------------------

    ObViSlamAdapter::ObViSlamAdapter(const std::string& config_file)
        : next_feature_id_(0) {
        vslam_types_refactor::readConfiguration(config_file, config_);
    }

    void ObViSlamAdapter::addKeyframe(
        vslam_types_refactor::FrameId frame_id,
        const std::vector<double>& pose_matrix,
        const std::vector<std::pair<double, double>>& keypoints,
        const std::vector<std::vector<unsigned char>>& /*descriptors*/,
        const std::unordered_map<vslam_types_refactor::CameraId,
        std::pair<int, int>>&image_sizes)
    {
        Eigen::Matrix4d mat;
        for (int i = 0; i < 16; ++i)
            mat(i / 4, i % 4) = pose_matrix[i];

        Eigen::Matrix3d R = mat.block<3, 3>(0, 0);
        Eigen::Vector3d t = mat.block<3, 1>(0, 3);
        vslam_types_refactor::Pose3D<double> pose;
        pose.transl_ = t;
        pose.orientation_ = Eigen::AngleAxisd(R);
        robot_poses_[frame_id] = pose;

        image_sizes_[frame_id] = image_sizes;

        const vslam_types_refactor::CameraId primary_camera_id = 0;
        for (size_t i = 0; i < keypoints.size(); ++i) {
            vslam_types_refactor::FeatureId fid = next_feature_id_++;

            vslam_types_refactor::StructuredVisionFeatureTrack track;
            track.feature_pos_ = vslam_types_refactor::Position3d<double>::Zero();

            vslam_types_refactor::VisionFeature obs(
                frame_id,
                { {primary_camera_id,
                  vslam_types_refactor::PixelCoord<double>(keypoints[i].first,
                                                           keypoints[i].second)} },
                primary_camera_id);
            track.feature_track.feature_observations_.emplace(frame_id, obs);
            visual_features_[fid] = track;
        }
    }

    void ObViSlamAdapter::setCameraIntrinsics(
        const std::unordered_map<vslam_types_refactor::CameraId,
        vslam_types_refactor::CameraIntrinsicsMat<double>>&intrinsics)
    {
        intrinsics_ = intrinsics;
    }

    void ObViSlamAdapter::setCameraExtrinsics(
        const std::unordered_map<vslam_types_refactor::CameraId,
        vslam_types_refactor::CameraExtrinsics<double>>&extrinsics)
    {
        extrinsics_ = extrinsics;
    }

    void ObViSlamAdapter::addDetections(
        vslam_types_refactor::FrameId frame_id,
        const std::unordered_map<vslam_types_refactor::CameraId,
        std::vector<vslam_types_refactor::RawBoundingBox>>&detections)
    {
        bounding_boxes_[frame_id] = detections;
    }

    bool ObViSlamAdapter::optimize() {
        if (intrinsics_.empty() || extrinsics_.empty()) {
            LOG(ERROR) << "Camera intrinsics or extrinsics not set. "
                << "Call setCameraIntrinsics() and setCameraExtrinsics() before optimize().";
            return false;
        }

        using namespace vslam_types_refactor;

        MainProbData prob_data(intrinsics_,
            extrinsics_,
            visual_features_,
            robot_poses_,
            config_.shape_dimension_priors_.mean_and_cov_by_semantic_class_,
            bounding_boxes_,
            long_term_map_,
            {});   // images not used

        auto continue_opt_checker = []() { return true; };

        FrameId max_frame_id = prob_data.getMaxFrameId();
        auto window_provider_func = [&](FrameId frame_id) -> FrameId {
            FrameId window_start = 0;
            if (frame_id > config_.sliding_window_params_.local_ba_window_size_)
                window_start = frame_id - config_.sliding_window_params_.local_ba_window_size_;
            return window_start;
            };

        using MainFactorInfo = std::pair<FactorType, FeatureFactorId>;
        auto refresh_residual_checker = [](const MainFactorInfo&,
            const MainPgPtr&,
            const util::EmptyStruct&) { return true; };

        IndependentEllipsoidsLongTermObjectMapFactorCreator<util::EmptyStruct, util::EmptyStruct>
            ltm_factor_creator(long_term_map_);

        auto cached_info_creator = [](const MainFactorInfo&,
            const MainPgPtr&,
            util::EmptyStruct&) { return true; };

        auto long_term_map_residual_creator_func =
            [&](const MainFactorInfo& factor_info,
                const MainPgPtr& pose_graph,
                const pose_graph_optimization::ObjectVisualPoseGraphResidualParams& residual_params,
                const std::function<bool(const MainFactorInfo&,
                    const MainPgPtr&,
                    util::EmptyStruct&)>& cached_info_create,
                ceres::Problem* problem,
                ceres::ResidualBlockId& res_id,
                util::EmptyStruct& cached_inf) -> bool {
                    return ltm_factor_creator.createResidual(factor_info,
                        pose_graph,
                        residual_params,
                        cached_info_create,
                        problem,
                        res_id,
                        cached_inf);
            };

        std::function<bool(
            const MainFactorInfo&,
            const MainPgPtr&,
            const pose_graph_optimization::ObjectVisualPoseGraphResidualParams&,
            const std::function<bool(const MainFactorInfo&,
                const MainPgPtr&,
                util::EmptyStruct&)>&,
            ceres::Problem*,
            ceres::ResidualBlockId&,
            util::EmptyStruct&)> ltm_res_func = long_term_map_residual_creator_func;

        std::function<bool(const MainFactorInfo&,
            const MainPgPtr&,
            util::EmptyStruct&)> cached_info_func = cached_info_creator;

        auto residual_creator = generateResidualCreator(ltm_res_func, cached_info_func);

        auto non_debug_residual_creator =
            [&](const MainFactorInfo& factor_id,
                const pose_graph_optimization::ObjectVisualPoseGraphResidualParams& solver_residual_params,
                const MainPgPtr& pose_graph,
                ceres::Problem* problem,
                ceres::ResidualBlockId& residual_id,
                util::EmptyStruct& cached_info) -> bool {
                    return residual_creator(factor_id,
                        solver_residual_params,
                        pose_graph,
                        false,
                        problem,
                        residual_id,
                        cached_info);
            };

        std::function<void(const MainProbData&, MainPgPtr&)> pose_graph_creator =
            [&](const MainProbData& input_data, MainPgPtr& pg) {
            std::unordered_map<ObjectId, std::pair<std::string, RawEllipsoid<double>>> ltm_objects;
            EllipsoidResults ellipsoids_in_map;
            if (input_data.getLongTermObjectMap() != nullptr) {
                input_data.getLongTermObjectMap()->getEllipsoidResults(ellipsoids_in_map);
            }
            for (const auto& e : ellipsoids_in_map.ellipsoids_) {
                ltm_objects[e.first] = std::make_pair(e.second.first,
                    convertToRawEllipsoid(e.second.second));
            }
            std::function<bool(const std::unordered_set<ObjectId>&,
                util::BoostHashMap<MainFactorInfo,
                std::unordered_set<ObjectId>>&)>
                ltm_factor_provider = [&](const std::unordered_set<ObjectId>& objects,
                    util::BoostHashMap<MainFactorInfo,
                    std::unordered_set<ObjectId>>&factor_data) -> bool {
                        return ltm_factor_creator.getFactorsToInclude(objects, factor_data);
                };
            pg = std::make_shared<MainPg>(input_data.getObjDimMeanAndCovByClass(),
                input_data.getCameraExtrinsicsByCamera(),
                input_data.getCameraIntrinsicsByCamera(),
                ltm_objects,
                ltm_factor_provider);
            };

        auto reprojection_error_provider = [&](const MainProbData& /*prob*/,
            const MainPgPtr& /*pg*/,
            FrameId /*frame_id*/,
            FeatureId /*feat_id*/,
            CameraId /*cam_id*/) -> double {
                return config_.visual_feature_params_.reprojection_error_std_dev_;
            };

        VisualFeatureFrontend<MainProbData> visual_feature_frontend(
            [](FrameId) { return false; },
            reprojection_error_provider,
            config_.visual_feature_params_.min_visual_feature_parallax_pixel_requirement_,
            config_.visual_feature_params_.min_visual_feature_parallax_robot_transl_requirement_,
            config_.visual_feature_params_.min_visual_feature_parallax_robot_orient_requirement_,
            config_.visual_feature_params_.enforce_min_pixel_parallax_requirement_,
            config_.visual_feature_params_.enforce_min_robot_pose_parallax_requirement_,
            config_.visual_feature_params_.inlier_epipolar_err_thresh_,
            config_.visual_feature_params_.check_past_n_frames_for_epipolar_err_,
            config_.visual_feature_params_.enforce_epipolar_error_requirement_,
            config_.visual_feature_params_.early_votes_return_,
            config_.visual_feature_params_.visual_feature_inlier_majority_percentage_);

        auto visual_feature_frame_data_adder = [&](const MainProbData& prob,
            const MainPgPtr& pg,
            FrameId min_frame,
            FrameId max_frame) {
                visual_feature_frontend.addVisualFeatureObservations(prob, pg, min_frame, max_frame);
            };

        std::function<Covariance<double, 6>(const Pose3D<double>&)> pose_dev_cov_func =
            [&](const Pose3D<double>& rel_pose) -> Covariance<double, 6> {
            return generateOdomCov(rel_pose,
                config_.object_visual_pose_graph_residual_params_.relative_pose_cov_params_.transl_error_mult_for_transl_error_,
                config_.object_visual_pose_graph_residual_params_.relative_pose_cov_params_.transl_error_mult_for_rot_error_,
                config_.object_visual_pose_graph_residual_params_.relative_pose_cov_params_.rot_error_mult_for_transl_error_,
                config_.object_visual_pose_graph_residual_params_.relative_pose_cov_params_.rot_error_mult_for_rot_error_);
            };

        auto bb_retriever = [&](FrameId frame_id,
            std::unordered_map<CameraId, std::vector<RawBoundingBox>>& bbs) -> bool {
                if (bounding_boxes_.find(frame_id) != bounding_boxes_.end()) {
                    bbs = bounding_boxes_[frame_id];
                    return true;
                }
                return false;
            };

        // Structures for bounding box front end
        auto all_observed_corner_locations_with_uncertainty =
            std::make_shared<std::unordered_map<FrameId,
            std::unordered_map<CameraId,
            std::vector<std::pair<BbCornerPair<double>,
            std::optional<double>>>>>>();
        auto associated_observed_corner_locations =
            std::make_shared<std::unordered_map<FrameId,
            std::unordered_map<CameraId,
            std::unordered_map<ObjectId,
            std::pair<BbCornerPair<double>,
            std::optional<double>>>>>>();
        auto bounding_boxes_for_pending_object =
            std::make_shared<std::vector<std::unordered_map<FrameId,
            std::unordered_map<CameraId,
            std::pair<BbCornerPair<double>, double>>>>>();
        auto pending_objects =
            std::make_shared<std::vector<std::pair<std::string,
            std::optional<EllipsoidState<double>>>>>();

        std::unordered_map<CameraId, std::pair<double, double>> img_heights_and_widths;
        for (const auto& [fid, sizes] : image_sizes_) {
            for (const auto& [cam_id, hw] : sizes) {
                img_heights_and_widths[cam_id] = { hw.first, hw.second };
            }
        }

        std::unordered_map<ObjectId, util::EmptyStruct> long_term_map_front_end_data;

        FeatureBasedBoundingBoxFrontEndCreator<ReprojectionErrorFactor> feature_based_associator_creator =
            generateFeatureBasedBbCreator(associated_observed_corner_locations,
                all_observed_corner_locations_with_uncertainty,
                bounding_boxes_for_pending_object,
                pending_objects,
                img_heights_and_widths,
                config_.bounding_box_covariance_generator_params_,
                config_.bounding_box_front_end_params_.geometric_similarity_scorer_params_,
                config_.bounding_box_front_end_params_.feature_based_bb_association_params_,
                long_term_map_front_end_data);

        auto bb_associator_retriever = [&](const MainPgPtr& pg, const MainProbData& /*prob*/) {
            return feature_based_associator_creator.getDataAssociator(pg);
            };

        // Build low‑level features map
        std::unordered_map<FrameId,
            std::unordered_map<CameraId,
            std::unordered_map<FeatureId, PixelCoord<double>>>>
            low_level_features_map;
        for (const auto& [fid, track] : visual_features_) {
            for (const auto& [frame, obs] : track.feature_track.feature_observations_) {
                for (const auto& [cam, pixel] : obs.pixel_by_camera_id) {
                    low_level_features_map[frame][cam][fid] = pixel;
                }
            }
        }

        using RawBoundingBoxContextInfo = FeatureBasedContextInfo;

        auto bb_context_retriever = [&](FrameId frame_id,
            CameraId cam_id,
            const MainProbData& /*prob*/)
            -> std::pair<bool, RawBoundingBoxContextInfo> {
            RawBoundingBoxContextInfo context;
            if (low_level_features_map.find(frame_id) != low_level_features_map.end() &&
                low_level_features_map.at(frame_id).find(cam_id) != low_level_features_map.at(frame_id).end()) {
                context.observed_features_ = low_level_features_map.at(frame_id).at(cam_id);
                return { true, context };
            }
            return { false, context };
            };

        // Manual frame addition (replaces addFrameDataAssociatedBoundingBox)
        auto frame_data_adder = [&](const MainProbData& prob,
            const MainPgPtr& pg,
            FrameId min_frame,
            FrameId max_frame) {
                // 1. Add frame to pose graph
                Pose3D<double> pose_at_frame_init_est;
                if (!prob.getRobotPoseEstimateForFrame(max_frame, pose_at_frame_init_est)) {
                    LOG(ERROR) << "Could not find initial estimate for frame " << max_frame;
                    return;
                }
                if (max_frame == 0) {
                    pg->addFrame(max_frame, pose_at_frame_init_est);
                }
                else {
                    Pose3D<double> prev_pose_init;
                    if (!prob.getRobotPoseEstimateForFrame(max_frame - 1, prev_pose_init)) {
                        LOG(ERROR) << "Could not find initial estimate for frame " << max_frame - 1;
                        pg->addFrame(max_frame, pose_at_frame_init_est);
                    }
                    else {
                        std::optional<RawPose3d<double>> revised_prev_pose_raw = pg->getRobotPose(max_frame - 1);
                        if (revised_prev_pose_raw.has_value()) {
                            Pose3D<double> relative_pose = getPose2RelativeToPose1(prev_pose_init, pose_at_frame_init_est);
                            Pose3D<double> corrected_pose = combinePoses(convertToPose3D(revised_prev_pose_raw.value()),
                                relative_pose);
                            pg->addFrame(max_frame, corrected_pose);
                        }
                        else {
                            pg->addFrame(max_frame, pose_at_frame_init_est);
                        }
                    }
                }

                // 2. Relative pose factors
                addConsecutiveRelativePoseFactorsForFrame(prob, pg, min_frame, max_frame, pose_dev_cov_func);

                // 3. Visual features
                visual_feature_frame_data_adder(prob, pg, min_frame, max_frame);

                // 4. Bounding boxes
                std::unordered_map<CameraId, std::vector<RawBoundingBox>> bounding_boxes_for_frame;
                if (bb_retriever(max_frame, bounding_boxes_for_frame)) {
                    auto bb_associator = bb_associator_retriever(pg, prob);
                    for (const auto& [cam_id, bbs] : bounding_boxes_for_frame) {
                        auto context = bb_context_retriever(max_frame, cam_id, prob);
                        if (context.first) {
                            bb_associator->addBoundingBoxObservations(max_frame,
                                cam_id,
                                bbs,
                                context.second);
                        }
                    }
                }
            };

        // Output data extractor
        CovarianceExtractorParams ltm_cov_params;
        IndependentEllipsoidsLongTermObjectMapExtractor<util::EmptyStruct> ltm_extractor(
            ltm_cov_params,
            residual_creator,
            [](const FactorType&, const FeatureFactorId&, ObjectId&) { return false; },
            config_.ltm_tunable_params_,
            config_.ltm_solver_residual_params_,
            config_.ltm_solver_params_);

        auto front_end_map_data_extractor = [&](std::unordered_map<ObjectId, util::EmptyStruct>& front_end_data) {
            return feature_based_associator_creator.getDataAssociator(nullptr)->getFrontEndObjMapData(front_end_data);
            };

        auto long_term_object_map_extractor = [&](const MainPgPtr& pg,
            const pose_graph_optimizer::OptimizationFactorsEnabledParams& params,
            MainLtm& ltm) -> bool {
                return ltm_extractor.extractLongTermObjectMap(pg, params, front_end_map_data_extractor, "", std::nullopt, ltm);
            };

        auto output_data_extractor = [&](const MainProbData& /*prob*/,
            const MainPgPtr& pg,
            const pose_graph_optimizer::OptimizationFactorsEnabledParams& params,
            LongTermObjectMapAndResults<MainLtm>& out) {
                extractLongTermObjectMapAndResults<MainLtm>(pg, params, long_term_object_map_extractor, out);
            };

        auto ceres_callback_creator = [](const MainProbData&, const MainPgPtr&, FrameId, FrameId) {
            return std::vector<std::shared_ptr<ceres::IterationCallback>>{};
            };

        auto visualization_callback = [](const MainProbData&, const MainPgPtr&,
            FrameId, FrameId,
            const VisualizationTypeEnum&, int) {};

        auto solver_params_provider_func = [&](FrameId frame_id)
            -> pose_graph_optimization::OptimizationIterationParams {
            if (frame_id == max_frame_id) return config_.final_ba_iteration_params_;
            if (frame_id % config_.sliding_window_params_.global_ba_frequency_ == 0)
                return config_.global_ba_iteration_params_;
            return config_.local_ba_iteration_params_;
            };

        auto merge_decider = [&](const MainPgPtr& pg,
            std::unordered_map<ObjectId, std::unordered_set<ObjectId>>& merge_results) -> bool {
                identifyMergeObjectsBasedOnCenterProximity(pg,
                    config_.bounding_box_front_end_params_.post_session_object_merge_params_.max_merge_distance_,
                    config_.bounding_box_front_end_params_.post_session_object_merge_params_.x_y_only_merge_,
                    merge_results);
                return true;
            };

        auto object_merger = [&](const MainPgPtr& pg) -> bool {
            std::unordered_map<ObjectId, std::unordered_set<ObjectId>> merge_results;
            if (!merge_decider(pg, merge_results)) return false;
            if (merge_results.empty()) return false;
            auto front_end = bb_associator_retriever(pg, prob_data);
            return front_end->mergeObjects(merge_results) && pg->mergeObjects(merge_results);
            };

        auto gba_checker = [&](FrameId frame_id) -> bool {
            FrameId window_start = window_provider_func(frame_id);
            return (frame_id - window_start > config_.sliding_window_params_.local_ba_window_size_);
            };

        OfflineProblemRunner<MainProbData,
            ReprojectionErrorFactor,
            LongTermObjectMapAndResults<MainLtm>,
            util::EmptyStruct,
            MainPg>
            runner(config_.object_visual_pose_graph_residual_params_,
                config_.limit_traj_eval_params_,
                config_.pgo_solver_params_,
                continue_opt_checker,
                window_provider_func,
                refresh_residual_checker,
                non_debug_residual_creator,
                pose_graph_creator,
                frame_data_adder,
                output_data_extractor,
                ceres_callback_creator,
                visualization_callback,
                solver_params_provider_func,
                object_merger,
                gba_checker);

        std::optional<OptimizationLogger> opt_logger;
        return runner.runOptimization(prob_data,
            config_.optimization_factors_enabled_params_,
            opt_logger,
            results_,
            0,
            true);
    }

    std::unordered_map<vslam_types_refactor::FrameId, vslam_types_refactor::Pose3D<double>>
        ObViSlamAdapter::getOptimizedTrajectory() const {
        return results_.robot_pose_results_.robot_poses_;
    }

    std::unordered_map<vslam_types_refactor::ObjectId,
        std::pair<std::string, vslam_types_refactor::EllipsoidState<double>>>
        ObViSlamAdapter::getObjectMap() const {
        return results_.ellipsoid_results_.ellipsoids_;
    }

} // namespace obvi_adapter


NB_MODULE(_obvi_slam, m) {
    m.doc() = "Python bindings for ObVi‑SLAM adapter (ROS‑free version)";

    // ------------------------------------------------------------------------
    // Bindings for ObViSlamAdapter
    // ------------------------------------------------------------------------
    nb::class_<obvi_adapter::ObViSlamAdapter>(m, "ObViSlamAdapter",
        R"doc(Adapter class for running ObVi‑SLAM without ROS.

        This class provides a ROS‑free interface to the ObVi‑SLAM pipeline,
        allowing direct injection of keyframes, detections, and camera parameters.)doc")
        .def(nb::init<const std::string&>(), nb::arg("config_file"),
            R"doc(Initialize the adapter with a configuration file.

            Args:
                config_file (str): Path to the YAML configuration file.
            )doc")
        .def("add_keyframe",
            [](obvi_adapter::ObViSlamAdapter& self,
                int frame_id,
                nb::ndarray<double, nb::shape<4, 4>, nb::c_contig> pose_mat,
                nb::ndarray<double, nb::shape<-1, 2>, nb::c_contig> keypoints,
                nb::ndarray<uint8_t, nb::shape<-1, 32>, nb::c_contig> descriptors,
                const std::unordered_map<int, std::pair<int, int>>& image_sizes) {
                    // Convert pose matrix to vector
                    const double* data = pose_mat.data();
                    std::vector<double> pose_vec(data, data + 16);
                    // Convert keypoints to vector of (x,y) pairs
                    std::vector<std::pair<double, double>> kps_vec;
                    kps_vec.reserve(keypoints.shape(0));
                    const double* kp_data = keypoints.data();
                    for (size_t i = 0; i < keypoints.shape(0); ++i) {
                        kps_vec.emplace_back(kp_data[2 * i], kp_data[2 * i + 1]);
                    }
                    // Convert descriptors to vector of vectors
                    std::vector<std::vector<unsigned char>> desc_vec;
                    desc_vec.reserve(descriptors.shape(0));
                    const uint8_t* desc_data = descriptors.data();
                    for (size_t i = 0; i < descriptors.shape(0); ++i) {
                        const uint8_t* row = desc_data + i * 32;
                        desc_vec.emplace_back(row, row + 32);
                    }
                    // Convert int keys to CameraId
                    std::unordered_map<vslam_types_refactor::CameraId,
                        std::pair<int, int>> image_sizes_cpp;
                    for (const auto& [cam_id, size] : image_sizes) {
                        image_sizes_cpp[static_cast<vslam_types_refactor::CameraId>(cam_id)] = size;
                    }
                    self.addKeyframe(frame_id, pose_vec, kps_vec, desc_vec, image_sizes_cpp);
            },
            nb::arg("frame_id"),
            nb::arg("pose_matrix").noconvert(),
            nb::arg("keypoints").noconvert(),
            nb::arg("descriptors").noconvert(),
            nb::arg("image_sizes"),
            R"doc(Add a keyframe with pose and ORB features.

            Args:
                frame_id (int): Unique identifier for the frame.
                pose_matrix (numpy.ndarray[float, shape=(4,4)]): 4x4 transformation matrix
                    from camera to world (row‑major).
                keypoints (numpy.ndarray[float, shape=(N,2)]): N keypoint coordinates (x,y) in pixels.
                descriptors (numpy.ndarray[uint8, shape=(N,32)]): N ORB descriptors (32 bytes each).
                image_sizes (dict[int, tuple[int,int]]): Mapping from camera ID to (width, height).
            )doc")
        .def("add_detections", &obvi_adapter::ObViSlamAdapter::addDetections,
            nb::arg("frame_id"), nb::arg("detections"),
            R"doc(Add object detections for a frame.

            Args:
                frame_id (int): Identifier of the frame to which detections belong.
                detections (list of RawBoundingBox): List of detected bounding boxes.
            )doc")
        .def("set_camera_intrinsics",
            [](obvi_adapter::ObViSlamAdapter& self,
                const std::unordered_map<int, nb::ndarray<double, nb::shape<3, 3>, nb::c_contig>>& intrinsics) {
                    std::unordered_map<vslam_types_refactor::CameraId,
                    vslam_types_refactor::CameraIntrinsicsMat<double>> intrinsics_cpp;
    for (const auto& [cam_id, mat] : intrinsics) {
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> map(mat.data());
        intrinsics_cpp[static_cast<vslam_types_refactor::CameraId>(cam_id)] = map;
    }
    self.setCameraIntrinsics(intrinsics_cpp);
            },
            nb::arg("intrinsics"),
            R"doc(Set camera intrinsics (3x3 matrices).

            Args:
                intrinsics (dict[int, numpy.ndarray[float, shape=(3,3)]]): Mapping from camera ID
                    to 3x3 intrinsic matrix (row‑major).
            )doc")
        .def("set_camera_extrinsics",
            [](obvi_adapter::ObViSlamAdapter& self,
                const std::unordered_map<int, nb::ndarray<double, nb::shape<4, 4>, nb::c_contig>>& extrinsics) {
                    std::unordered_map<vslam_types_refactor::CameraId,
                    vslam_types_refactor::CameraExtrinsics<double>> extrinsics_cpp;
    for (const auto& [cam_id, mat] : extrinsics) {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> map(mat.data());
        Eigen::Matrix3d R = map.block<3, 3>(0, 0);
        Eigen::Vector3d t = map.block<3, 1>(0, 3);
        vslam_types_refactor::Pose3D<double> pose;
        pose.transl_ = t;
        pose.orientation_ = Eigen::AngleAxisd(R);
        extrinsics_cpp[static_cast<vslam_types_refactor::CameraId>(cam_id)] = pose;
    }
    self.setCameraExtrinsics(extrinsics_cpp);
            },
            nb::arg("extrinsics"),
            R"doc(Set camera extrinsics (4x4 transformation matrices from camera to body).

            Args:
                extrinsics (dict[int, numpy.ndarray[float, shape=(4,4)]]): Mapping from camera ID
                    to 4x4 transformation matrix (row‑major) that maps points from camera frame to
                    body (e.g., IMU) frame.
            )doc")
        .def("optimize", &obvi_adapter::ObViSlamAdapter::optimize,
            R"doc(Run the full optimization (bundle adjustment) on all accumulated data.

            This method performs joint optimization of camera trajectory and object ellipsoids.
            )doc")
        .def("get_optimized_trajectory", &obvi_adapter::ObViSlamAdapter::getOptimizedTrajectory,
            R"doc(Get the refined camera trajectory after optimization.

            Returns:
                list of dict: Each element contains 'frame_id' and 'pose' (Pose3D object).
            )doc")
        .def("get_object_map", &obvi_adapter::ObViSlamAdapter::getObjectMap,
            R"doc(Get the refined object ellipsoid map after optimization.

            Returns:
                dict[int, EllipsoidState]: Mapping from object ID to its ellipsoid state.
            )doc");

    // ------------------------------------------------------------------------
    // Auxiliary types
    // ------------------------------------------------------------------------
    nb::class_<vslam_types_refactor::Pose3D<double>>(m, "Pose3D",
        R"doc(Represents a 3D pose (translation + rotation) using translation vector and angle‑axis.

        The rotation is stored as an Eigen::AngleAxisd (axis‑angle representation).
        )doc")
        .def(nb::init<>())
        .def_prop_ro("translation", [](const vslam_types_refactor::Pose3D<double>& p) {
        return nb::make_tuple(p.transl_.x(), p.transl_.y(), p.transl_.z());
            }, R"doc(Translation vector (x, y, z).)doc")
        .def_prop_ro("rotation_quaternion", [](const vslam_types_refactor::Pose3D<double>& p) {
        Eigen::Quaterniond q(p.orientation_.toRotationMatrix());
        return nb::make_tuple(q.w(), q.x(), q.y(), q.z());
            }, R"doc(Rotation as quaternion (w, x, y, z).)doc")
        .def_prop_ro("rotation_matrix", [](const vslam_types_refactor::Pose3D<double>& p) {
        return p.orientation_.toRotationMatrix().cast<double>();
            }, R"doc(Rotation matrix (3x3).)doc");

    nb::class_<vslam_types_refactor::EllipsoidState<double>>(m, "EllipsoidState",
        R"doc(State of an ellipsoidal object: 3D pose + dimensions (semi‑axes lengths).)doc")
        .def(nb::init<>())
        .def_ro("pose", &vslam_types_refactor::EllipsoidState<double>::pose_,
            R"doc(Pose of the ellipsoid center (Pose3D object).)doc")
        .def_prop_ro("dimensions", [](const vslam_types_refactor::EllipsoidState<double>& e) {
        return nb::make_tuple(e.dimensions_.x(), e.dimensions_.y(), e.dimensions_.z());
            }, R"doc(Semi‑axes lengths (dx, dy, dz).)doc");

    nb::class_<vslam_types_refactor::RawBoundingBox>(m, "RawBoundingBox",
        R"doc(Represents a raw bounding box detection from an object detector.)doc")
        .def(nb::init<>())
        .def_rw("pixel_corner_locations",
            &vslam_types_refactor::RawBoundingBox::pixel_corner_locations_,
            R"doc(Pair of corner coordinates: (min_x, min_y) and (max_x, max_y))doc")
        .def_rw("semantic_class",
            &vslam_types_refactor::RawBoundingBox::semantic_class_,
            R"doc(Semantic class string (e.g., 'car', 'person').)doc")
        .def_rw("detection_confidence",
            &vslam_types_refactor::RawBoundingBox::detection_confidence_,
            R"doc(Detection confidence score in [0,1].)doc");

    nb::class_<vslam_types_refactor::BbCornerPair<double>>(m, "BbCornerPair",
        R"doc(Pair of two PixelCoord representing bounding box corners.)doc")
        .def(nb::init<>())
        .def_rw("first", &vslam_types_refactor::BbCornerPair<double>::first,
            R"doc(First corner (top‑left).)doc")
        .def_rw("second", &vslam_types_refactor::BbCornerPair<double>::second,
            R"doc(Second corner (bottom‑right).)doc");
}