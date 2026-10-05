"""Pre-collection teacher checks, independent of dataset completion."""

PRECOLLECTION_CASES = ("feasibility", "calibration", "teacher", "recovery", "convergence", "prior_revision", "replay", "viewer", "regressions")


def local_teacher_passed(report):
    if any(report.get("cases", {}).get(name, {}).get("status") != "passed" for name in PRECOLLECTION_CASES):
        return False
    normal, recovery = report.get("baseline", {}), report.get("recovery_baseline", {})
    if report.get("robot_id") == "panda":
        return (normal.get("attempts") == 100 and normal.get("successes", 0) >= 95
                and recovery.get("attempts") == 10 and recovery.get("successes") == 10)
    return (report.get("robot_id") == "ur5e" and normal.get("attempts") == 5
            and normal.get("successes") == 5 and recovery.get("attempts") == 5
            and recovery.get("successes") == 5)


def panda_teacher_passed(report):
    return (report.get("robot_id") == "panda" and local_teacher_passed(report)
            and report.get("parent_m3", {}).get("status") == "passed")
