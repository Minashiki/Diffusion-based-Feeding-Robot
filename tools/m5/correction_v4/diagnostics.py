"""Record the exact evidence supplied to the unchanged pickup event logic."""

from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np

from feedingrobot.data.episodes import write_json

FIELDS=('tick','dt','supported','off_bowl','linear_speed','angular_speed','pickup_eligible',
    'bowl_clearance_m','timer_before','timer_after','reset_mask')


class PickupRecorder:
    def __init__(self): self.rows=[]

    @contextmanager
    def capture(self,task):
        logic=task.logic;original=logic.update
        owned='update' in logic.__dict__
        def update(e,dt,time,**kwargs):
            velocity=np.empty(6)
            mujoco.mj_objectVelocity(task.model,task.data,mujoco.mjtObj.mjOBJ_BODY,
                int(task.index.bean_bodies[0]),velocity,0)
            linear=float(np.linalg.norm(velocity[3:]));angular=float(np.linalg.norm(velocity[:3]))
            before=logic.timers['pickup']
            failed=[not e['supported'],not e['off_bowl'],
                linear>=task.bean_acceptance['linear_speed_m_s'],
                angular>=task.bean_acceptance['angular_speed_rad_s']]
            result=original(e,dt,time,**kwargs)
            self.rows.append([task.tick,dt,e['supported'],e['off_bowl'],linear,angular,
                e['pickup_eligible'],e['bowl_clearance_m'],before,logic.timers['pickup'],
                sum((1<<i) for i,bad in enumerate(failed) if bad)])
            return result
        logic.update=update
        try: yield
        finally:
            if owned: logic.update=original
            else: del logic.update

    def save(self,directory):
        rows=np.asarray(self.rows,dtype=np.float64).reshape(-1,len(FIELDS))
        np.save(Path(directory)/'pickup_boundaries.npy',rows)
        resets=rows[(rows[:,8]>0)&(rows[:,9]==0)]
        summary=dict(fields=FIELDS,rows=len(rows),zero_dt_boundaries=int(np.count_nonzero(rows[:,1]==0)),
            longest_qualified_s=float(rows[:,9].max(initial=0)),
            reset_causes={name:int(np.count_nonzero(resets[:,10].astype(int)&(1<<i)))
                for i,name in enumerate(('support','off_bowl','linear_speed','angular_speed'))},
            resets=[dict(tick=int(r[0]),prior_qualified_s=float(r[8]),mask=int(r[10])) for r in resets])
        write_json(Path(directory)/'pickup_diagnostics.json',summary)
        return summary


def replay_with_pickup(directory,output):
    from feedingrobot.data import replay
    original=replay.FeedingGymEnv;recorder=PickupRecorder()
    def environment(*args,**kwargs):
        env=original(*args,**kwargs);step=env.task.step_physics
        def observed_step(**options):
            if options.get('_settling') or env.task.logic is None: return step(**options)
            with recorder.capture(env.task): return step(**options)
        env.task.step_physics=observed_step
        return env
    replay.FeedingGymEnv=environment
    try: result=replay.replay_episode(directory)
    finally: replay.FeedingGymEnv=original
    return dict(replay=result,pickup=recorder.save(output))
