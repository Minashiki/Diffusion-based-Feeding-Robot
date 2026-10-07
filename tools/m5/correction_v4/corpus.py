"""Three pools through pickup confirmation, with bounded target windows."""

from pathlib import Path

import numpy as np
from torch.utils.data import default_collate

from feedingrobot.policies.data import ActionWindows
from feedingrobot.sim.model import ROOT

STAGES=('pre_entry','entry','sweep','wall_align','wall_45','wall_30','wall_15',
    'wall_level','clearance','wall_seat_tip','wall_seat_level','pickup_hold')


class PickupWindows(ActionWindows):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pools={p:{} for p in ('alignment','transition','aligned')}
        self.tags={};self.clipped=0
        for e,(path,m) in enumerate(self.episodes):
            owners=np.load(path/'action_owner.npy',allow_pickle=False)
            stages=np.load(path/'action_stages.npy',allow_pickle=False)
            mask=np.load(path/'action_mask.npy',allow_pickle=False)
            phases=np.load(path/'action_phases.npy',allow_pickle=False)
            ticks=self.arrays(e)['action_ticks']
            acquisition=(phases==1)|((phases==2)&(stages=='pickup_hold'))
            permitted=mask&acquisition&np.isin(owners,('alignment_teacher','path_teacher'))
            permitted&=(owners=='alignment_teacher')|np.isin(stages,STAGES)
            hold=np.flatnonzero(permitted&(stages=='pickup_hold'))
            if not len(hold): raise ValueError('Missing complete pickup_hold supervision')
            cutoff=hold[-1]+1
            permitted[cutoff:]=False
            for i,(episode,start,end,phase) in enumerate(self.windows):
                if episode!=e or not permitted[start]: continue
                bounded=start
                while bounded<end and permitted[bounded]: bounded+=1
                if bounded<end:
                    self.clipped+=1;self.windows[i]=(episode,start,bounded,phase)
                if owners[start]=='alignment_teacher':
                    if m['category']=='aligned': raise ValueError('Aligned control contains correction')
                    wait=m['handover'].get('stable_start_tick',m['corrected_tick']-200)
                    pool='alignment';stage='stable_wait' if ticks[start]>=wait else 'correcting'
                else:
                    pool='aligned' if m['category']=='aligned' else 'transition';stage=str(stages[start])
                    if pool=='transition' and ticks[start]<m['corrected_tick']:
                        raise ValueError('Path supervision precedes correction')
                self.pools[pool].setdefault(e,{}).setdefault(stage,[]).append(i)
                self.tags[i]=(pool,stage)
            required=('aligned',) if m['category']=='aligned' else ('alignment','transition')
            if any(e not in self.pools[p] for p in required): raise ValueError('Missing window pool')
            path_pool='aligned' if m['category']=='aligned' else 'transition'
            needed=set(STAGES)-({'pre_entry'} if m['category'].startswith('p2') else set())
            if not needed<=self.pools[path_pool][e].keys(): raise ValueError('Missing pickup path stage')
            if m['category']!='aligned' and set(self.pools['alignment'][e])!={'correcting','stable_wait'}:
                raise ValueError('Missing correction/wait supervision')

    def summary(self):
        unique={};positions={};starts={};probability={};intervals={}
        for pool,episodes in self.pools.items():
            for e,stages in episodes.items():
                ticks=self.arrays(e)['action_ticks']
                actual=np.load(self.episodes[e][0]/'action_stages.npy',allow_pickle=False)
                for stage,indices in stages.items():
                    probability.setdefault(pool,{})[f'{e}:{stage}']=1/12/len(episodes)/len(stages)
                    intervals[f'{pool}:{e}:{stage}']=[int(min(ticks[self.windows[i][1]] for i in indices)),
                        int(max(ticks[self.windows[i][1]] for i in indices))]
                    starts[stage]=starts.get(stage,0)+len(indices)
                    for i in indices:
                        _,start,end,_=self.windows[i]
                        for row in range(start,end):
                            name=str(actual[row]);unique.setdefault(name,set()).add((e,row))
                            positions[name]=positions.get(name,0)+1
        return dict(pools={p:dict(episodes=len(rows),windows=sum(len(ids) for stages in rows.values()
            for ids in stages.values())) for p,rows in self.pools.items()},
            starts_by_stage=starts,unique_actions_by_stage={s:len(rows) for s,rows in unique.items()},
            loss_positions_by_stage=positions,clipped_windows=self.clipped,
            stage_probabilities=probability,intervals=intervals,
            sampling='75% original; 25% equal pool, then episode/stage/window balanced',
            correction_fraction=.25,pool_probability=1/12)


class MixedV4:
    def __init__(self,config,normalization,directory):
        self.base=ActionWindows(ROOT/config['dataset'],'train',config['model']['horizon'],normalization)
        self.extra=PickupWindows(directory,'train',config['model']['horizon'],normalization)

    def batch(self,rng,count):
        if count<4 or count%4: raise ValueError('Batch size must be divisible by four')
        items=[self.base[i] for i in self.base.sample_indices(rng,count*3//4)]
        names=tuple(self.extra.pools)
        for _ in range(count//4):
            pool=self.extra.pools[names[int(rng.integers(3))]]
            episodes=sorted(pool);e=episodes[int(rng.integers(len(episodes)))]
            stages=sorted(pool[e]);stage=stages[int(rng.integers(len(stages)))]
            indices=pool[e][stage];items.append(self.extra[indices[int(rng.integers(len(indices)))]] )
        return default_collate([items[i] for i in rng.permutation(count)])
