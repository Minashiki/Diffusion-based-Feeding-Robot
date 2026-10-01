"""Contact truth and physical-substep aggregation, separate from wrist F/T."""

import mujoco
import numpy as np


def read_contacts(model, data, index):
    rows = []
    for i in range(data.ncon):
        contact = data.contact[i]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        rows.append(dict(geom1=int(contact.geom1), geom2=int(contact.geom2),
                         group1=index.group(model, contact.geom1), group2=index.group(model, contact.geom2),
                         force_on_geom2_world=force.copy(), force_n=float(np.linalg.norm(force)),
                         position=contact.pos.copy(), distance=float(contact.dist)))
    return rows


class ContactMonitor:
    def __init__(self, limit):
        self.limit = limit
        self.reset()

    def reset(self):
        self.peak_n = 0.
        self.last_peak_n = 0.
        self.impulse_ns = 0.
        self.over_limit_s = 0.
        self.events = []

    def update(self, contacts, dt, time):
        # Sum contacts per body-group pair; splitting a surface must not hide force.
        sums = {}
        for row in contacts:
            pair = tuple(sorted([row["group1"], row["group2"]]))
            sums[pair] = sums.get(pair, 0.) + row["force_n"]
        guarded = {p: f for p, f in sums.items() if ("spoon" in p or "arm" in p)}
        peak = max(guarded.values(), default=0.)
        self.last_peak_n = peak
        self.peak_n = max(self.peak_n, peak)
        self.impulse_ns += sum(guarded.values()) * dt
        if peak > self.limit:
            self.over_limit_s += dt
            self.events.append(dict(time=time, force_n=peak))
        return peak > self.limit
