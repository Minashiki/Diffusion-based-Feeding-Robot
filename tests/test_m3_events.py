"""Synthetic evidence tests validate logic, not physical feeding success."""

import numpy as np
import pytest

from feedingrobot.sim.events import TaskEvents
from feedingrobot.sim.model import load_json


def sample(**overrides):
    row = dict(supported=False, off_bowl=False, on_bowl=True, mouth_supported=False, released=True,
               tool_inside=False, tool_mouth_contact=False, at_wait=False, aligned=True, ready=True,
               food_ground_contact=False, food_valid=True, penetration=False, bean_penetration=False, pickup_eligible=False,
               tcp_position=np.zeros(3), bean_position=np.ones(3), mouth_position=np.ones(3),
               wait_position=np.ones(3))
    row.update(overrides)
    row["pickup_eligible"] = row["supported"] and row["off_bowl"]
    return row


@pytest.fixture
def logic():
    return TaskEvents(load_json("configs/task.json"))


def advance(logic, e, seconds, failure=None):
    for _ in range(round(seconds / .001)):
        t = getattr(logic, "test_time", 0.) + .001
        logic.test_time = t
        logic.update(e, .001, t, failure=failure)
        if logic.failure_reason or logic.success:
            break


def acquired(logic):
    e = sample(on_bowl=False, off_bowl=True, supported=True, released=False)
    advance(logic, e, .501)
    assert logic.phase == "TRANSPORT"
    return e


def transferring(logic):
    e = acquired(logic)
    e["at_wait"] = True
    advance(logic, e, .101)
    assert logic.phase == "APPROACH"
    e.update(at_wait=False, tool_inside=True, tool_mouth_contact=True)
    advance(logic, e, .001)
    assert logic.phase == "TRANSFER"
    return e


def test_full_phase_sequence_and_continuous_receipt(logic):
    e = transferring(logic)
    # Merely arriving at the mouth, including mouth contact, cannot deliver food.
    e["mouth_supported"] = True
    advance(logic, e, .3)
    assert not logic.delivered
    e.update(supported=False, released=True)
    advance(logic, e, .199)
    assert not logic.delivered
    advance(logic, e, .001)
    assert logic.phase == "RETRACT" and logic.delivered
    advance(logic, e, .2)
    assert not logic.success
    e.update(tool_inside=False, tool_mouth_contact=False)
    advance(logic, e, .099)
    assert not logic.success
    advance(logic, e, .001)
    assert logic.success
    assert [r["phase"] for r in logic.events if r["name"] == "phase"] == [
        "ACQUIRE", "TRANSPORT", "WAIT_READY", "APPROACH", "TRANSFER", "RETRACT"]


def test_pickup_needs_contact_leave_plate_and_continuity(logic):
    e = sample(supported=True)
    advance(logic, e, .2)
    assert not logic.acquired
    e.update(off_bowl=True, on_bowl=False)
    e["pickup_eligible"] = True
    advance(logic, e, .499)
    assert not logic.acquired
    e["supported"] = False
    advance(logic, e, .001)
    e["supported"] = True
    e["pickup_eligible"] = True
    advance(logic, e, .499)
    assert not logic.acquired
    advance(logic, e, .001)
    assert logic.acquired


def test_receipt_timer_resets_and_retract_needs_no_contact(logic):
    e = transferring(logic)
    e.update(supported=False, released=True, mouth_supported=True)
    advance(logic, e, .199)
    e["released"] = False
    advance(logic, e, .001)
    e["released"] = True
    advance(logic, e, .199)
    assert not logic.delivered
    advance(logic, e, .001)
    e.update(tool_inside=False)
    advance(logic, e, .2)
    assert not logic.success
    e["tool_mouth_contact"] = False
    advance(logic, e, .1)
    assert logic.success


def test_wait_and_recovery_do_not_generate_motion(logic):
    e = acquired(logic)
    e.update(at_wait=True, ready=False)
    advance(logic, e, .2)
    assert logic.phase == "WAIT_READY"
    e["ready"] = True
    advance(logic, e, .1)
    assert logic.phase == "APPROACH"
    e.update(at_wait=False, ready=False)
    advance(logic, e, .001)
    assert logic.phase == "RECOVER"
    advance(logic, e, .2)
    assert logic.phase == "RECOVER"
    e["at_wait"] = True
    advance(logic, e, .001)
    assert logic.phase == "WAIT_READY"


@pytest.mark.parametrize("kind", ["unsupported", "floor", "back_on_bowl", "early_withdrawal", "post_delivery_loss"])
def test_distinct_food_failures(logic, kind):
    e = transferring(logic)
    if kind == "early_withdrawal":
        e["tool_inside"] = False
        expected = "withdrawal_before_release"
    elif kind == "post_delivery_loss":
        e.update(supported=False, released=True, mouth_supported=True)
        advance(logic, e, .2)
        e["mouth_supported"] = False
        expected = "food_lost_after_delivery"
    else:
        e.update(supported=False, released=True)
        e["food_ground_contact"] = kind == "floor"
        e["on_bowl"] = kind == "back_on_bowl"
        expected = "food_dropped"
    advance(logic, e, .1)
    assert logic.failure_reason == expected and not logic.success


def test_transfer_grace_and_no_empty_delivery(logic):
    e = transferring(logic)
    e.update(supported=False, released=True)
    advance(logic, e, .05)
    assert logic.failure_reason is None
    e["mouth_supported"] = True
    advance(logic, e, .2)
    assert logic.delivered
    other = TaskEvents(logic.config)
    advance(other, sample(on_bowl=False, off_bowl=True, mouth_supported=True), .3)
    assert not other.acquired and not other.delivered


def test_penetration_duration_and_simultaneous_failure_wins(logic):
    e = sample(penetration=True)
    advance(logic, e, .004)
    assert logic.failure_reason is None
    e["penetration"] = False
    advance(logic, e, .001)
    e["penetration"] = True
    advance(logic, e, .005)
    assert logic.failure_reason == "model_penetration"
    other = TaskEvents(logic.config)
    e = transferring(other)
    e.update(supported=False, released=True, mouth_supported=True)
    advance(other, e, .2)
    e.update(tool_inside=False, tool_mouth_contact=False)
    advance(other, e, .099)
    advance(other, e, .001, failure="contact_limit")
    assert not other.success and other.failure_reason == "contact_limit"
    assert any(r["name"] == "success_candidate" for r in other.events)
    assert not any(r["name"] == "success" for r in other.events)


@pytest.mark.parametrize("failure", ["blocked", "ik_failure", "workspace_limit", "nonfinite_state", "joint_speed_limit"])
def test_hard_failure_preserved(logic, failure):
    advance(logic, sample(), .001, failure=failure)
    assert logic.failure_reason == failure


def test_invalid_target(logic):
    advance(logic, sample(food_valid=False), .001)
    assert logic.failure_reason == "food_missing"
