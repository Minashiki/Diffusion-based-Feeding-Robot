"""Independent Mink reachability checks, never applied to simulator qpos."""

import mink
import numpy as np

from feedingrobot.control.adapter import clip_norm


def check_waypoints(task, waypoints):
    configuration = mink.Configuration(task.model)
    configuration.update(task.data.qpos.copy())
    frame = mink.FrameTask(task.robot_config["tcp_site"], "site", position_cost=1., orientation_cost=1., lm_damping=1e-4)
    results = []
    for name, position, rotation in waypoints:
        target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rotation), np.asarray(position))
        frame.set_target(target)
        error = np.full(6, np.inf)
        try:
            for _ in range(1200):
                actual = configuration.get_transform_frame_to_world(task.robot_config["tcp_site"], "site")
                error = np.r_[target.translation()-actual.translation(),
                              (target.rotation() @ actual.rotation().inverse()).log()]
                if np.linalg.norm(error[:3]) <= .002 and np.linalg.norm(error[3:]) <= .02:
                    break
                # Continue through bounded TCP goals, as the execution adapter does,
                # instead of asking differential IK to jump to a remote frame.
                incremental = mink.SE3.from_rotation_and_translation(
                    mink.SO3.exp(clip_norm(error[3:], task.robot_config["angular_speed_limit"]*.01)) @ actual.rotation(),
                    actual.translation()+clip_norm(error[:3], task.robot_config["linear_speed_limit"]*.01))
                frame.set_target(incremental)
                velocity = mink.solve_ik(configuration, [frame], .01, solver="daqp", limits=task.adapter.limits,
                                         constraints=[task.adapter.freeze], damping=1e-5, safety_break=True)
                configuration.integrate_inplace(velocity, .01)
            results.append(dict(name=name, reachable=bool(np.linalg.norm(error[:3]) <= .002
                                                        and np.linalg.norm(error[3:]) <= .02),
                                position_error_m=float(np.linalg.norm(error[:3])), rotation_error_rad=float(np.linalg.norm(error[3:]))))
        except (mink.exceptions.MinkError, ValueError, np.linalg.LinAlgError) as exc:
            results.append(dict(name=name, reachable=False, reason=str(exc)))
    return results
