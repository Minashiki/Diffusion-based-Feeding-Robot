"""Recorded ellipsoid/mesh poses must push the bean above the scoop upward."""

import mujoco
import pytest

from feedingrobot.sim.model import ROOT, load_json


# Reduced from actual 65/75/70 degree scoop runs. Only one source prism and
# one ellipsoid remain; no motion, robot, bowl or other beans can change poses.
FIXTURES = [
    dict(
        bean_position=[0.4841240234688748, -0.17588458489578998, 0.07952979511793055],
        bean_quaternion=[0.614674300511803, -0.13774074031710856, 0.056221707869055566, -0.7746238521447087],
        spoon_position=[0.4417829365221396, -0.16732931956696737, 0.08693529887136858],
        spoon_quaternion=[0.9941380368438905, 0.005599091434903835, 0.06367329936420942, -0.08720048637108976],
        mesh='assets/task/tableware/spoon/meshes/scoop_prisms_coarse/spoon_scoop_coarse_prism_061.obj'),
    dict(
        bean_position=[0.4958366256732505, -0.17731526331701208, 0.03978830425158065],
        bean_quaternion=[-0.48742430796585423, -0.559964179393368, 0.5637363298635272, 0.3620207344784565],
        spoon_position=[0.4478596988050545, -0.16837251000667422, 0.06281150006437358],
        spoon_quaternion=[0.9677989738279298, 0.02068399696975387, 0.23610733710750587, -0.08479766441927396],
        mesh='assets/task/tableware/spoon/meshes/scoop_prisms_coarse/spoon_scoop_coarse_prism_104.obj'),
    dict(
        bean_position=[0.4940512951739818, -0.1773201468653853, 0.04747166895432055],
        bean_quaternion=[-0.29915863803705317, -0.6899887226692926, 0.6407077350250292, 0.15463916113362186],
        spoon_position=[0.4464937081954257, -0.16814602884059088, 0.06778115993568978],
        spoon_quaternion=[0.9747466480689706, 0.018025245665003575, 0.2055158274735196, -0.08548278923208413],
        mesh='assets/task/tableware/spoon/meshes/scoop_prisms_coarse/spoon_scoop_coarse_prism_105.obj'),
]


@pytest.mark.parametrize('fixture', FIXTURES, ids=['prism061', 'prism104', 'prism105'])
def test_contact_normal_points_out_of_spoon(fixture):
    scene = load_json('configs/scene.json')
    bean = scene['beans']
    def values(array):
        return ' '.join(str(value) for value in array)
    xml = f'''<mujoco><option ccd_tolerance="{scene['ccd_tolerance']}"/>
    <asset><mesh name="surface" file="{ROOT / fixture['mesh']}"/></asset>
    <worldbody><geom name="surface" type="mesh" mesh="surface"
      pos="{values(fixture['spoon_position'])}" quat="{values(fixture['spoon_quaternion'])}"/>
    <body pos="{values(fixture['bean_position'])}" quat="{values(fixture['bean_quaternion'])}">
      <freejoint/><geom name="bean" type="ellipsoid" size="{values(bean['semi_axes_m'])}"
        margin="{bean['margin']}" priority="{bean['priority']}"/>
    </body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    assert not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    contact = data.contact[0]
    sign = -1 if model.geom(int(contact.geom1)).name == 'bean' else 1
    normal = sign * contact.frame[:3]
    assert 0 < contact.dist < bean['margin']
    assert normal[2] > .8
