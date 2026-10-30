import numpy as np
import pytest
import torch
from utils.arm_normal_confidence import load_weights, normal_confidence, topology_digest

FACES=np.array([[0,1,2]],dtype=np.int64)
def bind(path,**kw):
 np.savez(path,faces_sha256=topology_digest(FACES),vertex_count=3,**kw)

def test_binding_ids_and_weights(tmp_path):
 p=tmp_path/'x.npz';bind(p,vertex_ids=np.array([0,2]));np.testing.assert_array_equal(load_weights(p,FACES,3),[1,0,1]);bind(p,weights=np.array([.2,0,1.]));np.testing.assert_allclose(load_weights(p,FACES,3),[.2,0,1.])

def test_refuse_wrong_binding_and_invalid_inputs(tmp_path):
 p=tmp_path/'x.npz';bind(p,vertex_ids=np.array([0,3]));
 with pytest.raises(ValueError):load_weights(p,FACES,3)
 bind(p,weights=np.array([0,np.nan,1]));
 with pytest.raises(ValueError):load_weights(p,FACES,3)
 bind(p,vertex_ids=np.array([0]));
 with pytest.raises(ValueError):load_weights(p,np.array([[0,2,1]]),3)
 with pytest.raises(ValueError):topology_digest(np.array([[0.,1.,2.]]))

class Renderer:
 def render(self,v,f,colors):
  out=torch.zeros((1,5,5,4));out[0,2,2,0]=1;return out

def test_confidence_dilates_and_respects_visible_alpha():
 v=torch.zeros((3,3));f=torch.tensor(FACES);state=torch.zeros((3,10));alpha=torch.ones((1,5,5,1));alpha[0,1,1,0]=0
 result=normal_confidence(Renderer(),v,f,state,alpha,dilation=1)
 assert result[0,2,2,0] == 0 and result[0,1,2,0] == 0
 assert result[0,1,1,0] == 1 and result[0,0,0,0] == 1

def test_outer_compaction_mapping_and_wrong_topology(tmp_path):
 from utils.arm_normal_confidence import remap_weights
 p=tmp_path/'full.npz';full=np.array([[0,1,2],[3,4,5]],np.int64);np.savez(p,faces_sha256=topology_digest(full),vertex_count=6,vertex_ids=np.array([1,4]))
 np.testing.assert_array_equal(remap_weights(p,full,6,np.array([0,1,2]),np.array([[0,1,2]])),[0,1,0])
 with pytest.raises(ValueError):remap_weights(p,full,6,np.array([0,1,2]),np.array([[0,2,1]]))
 with pytest.raises(ValueError):remap_weights(p,full,6,np.array([0,1,2]),np.array([[0,1,2],[0,1,2]]))
 with pytest.raises(ValueError):remap_weights(p,np.array([[0,2,1],[3,4,5]]),6,np.array([0,1,2]),np.array([[0,1,2]]))

def test_optional_known_label_remap_and_conflicts(tmp_path):
 from utils.arm_normal_confidence import remap_known_non_arm
 p=tmp_path/'x.npz';bind(p,vertex_ids=np.array([0]),known_non_arm_ids=np.array([2]))
 np.testing.assert_array_equal(remap_known_non_arm(p,FACES,3,np.arange(3),np.array([1,0,0])),[0,0,1])
 with pytest.raises(ValueError): remap_known_non_arm(p,FACES,3,np.arange(3),np.array([0,0,1]))
 bind(p,vertex_ids=np.array([0]));assert remap_known_non_arm(p,FACES,3,np.arange(3),np.array([1,0,0])) is None

def test_configurable_weight_spill_and_unknown():
 class ColoredRenderer:
  def render(self,v,f,colors):
   o=torch.zeros((1,5,5,4));o[0,2,2,0]=1;o[0,2,3,1]=1;return o
 v=torch.zeros((3,3));f=torch.tensor(FACES);state=torch.zeros((3,11));alpha=torch.ones((1,5,5,1));result,stats=normal_confidence(ColoredRenderer(),v,f,state,alpha,dilation=1,attenuation=.25,return_stats=True)
 assert result[0,2,2,0]==.25 and result[0,0,0,0]==1
 assert stats['maskedPixelsByView']==[9] and stats['knownNonArmSpillByView']==[1] and stats['unknownMaskedPixelsByView']==[7]
 for invalid in (-1.,1.1,float('nan'),True):
  with pytest.raises(ValueError): normal_confidence(ColoredRenderer(),v,f,state,alpha,attenuation=invalid)

def test_attach_preserves_extra_state_across_split():
 from utils.arm_normal_confidence import attach
 class Optimizer:
  def __init__(self):self._vertices_etc=torch.zeros((3,9))
  def _split_vertices_etc(self):self._vertices=self._vertices_etc[:,:3]
 o=Optimizer();attach(o,np.array([1,0,0]),np.array([0,0,1]));assert o._vertices_etc.shape==(3,11)
 copied=o._vertices_etc[[0,2,1]].detach().clone();o._vertices_etc=copied;o._split_vertices_etc()
 assert o._vertices_etc[:,9].tolist()==[1,0,0] and o._vertices_etc[:,10].tolist()==[0,1,0]

def test_refuse_fractional_and_bool_counts_and_complex_bool_weights(tmp_path):
 p=tmp_path/'x.npz'
 for count in (3.9,3.0,True):
  np.savez(p,faces_sha256=topology_digest(FACES),vertex_count=count,vertex_ids=np.array([0]))
  with pytest.raises(ValueError): load_weights(p,FACES,3)
 for weights in (np.array([1+2j,0j,0j]),np.array([True,False,False])):
  bind(p,weights=weights)
  with pytest.raises(ValueError):load_weights(p,FACES,3)

def test_reject_bad_threshold_and_nonfinite_transported_state():
 v=torch.zeros((3,3));f=torch.tensor(FACES);state=torch.zeros((3,10));alpha=torch.ones((1,5,5,1))
 for bad in (True,-.1,1.1,float('nan')):
  with pytest.raises(ValueError):normal_confidence(Renderer(),v,f,state,alpha,threshold=bad)
 state[0,9]=float('nan')
 with pytest.raises(ValueError):normal_confidence(Renderer(),v,f,state,alpha)

def test_attenuation_one_preserves_loss_and_alpha():
 v=torch.zeros((3,3));f=torch.tensor(FACES);state=torch.zeros((3,10));alpha=torch.ones((1,5,5,1));before=alpha.clone();res=normal_confidence(Renderer(),v,f,state,alpha,dilation=1,attenuation=1.0);assert torch.equal(res,torch.ones_like(res));assert torch.equal(alpha,before)

def test_capture_loaded_initial_state_is_exact_and_exclusive(tmp_path):
 from utils.arm_normal_confidence import capture_initial_state
 path=tmp_path/'state.npz';tensor=torch.arange(12,dtype=torch.float32).reshape(3,4)[:,::2]
 receipt=capture_initial_state(path,normal_target=tensor)
 with np.load(path,allow_pickle=False) as saved:np.testing.assert_array_equal(saved['normal_target'],tensor.numpy())
 assert receipt['arrays']['normal_target']['shape']==[3,2]
 with pytest.raises(FileExistsError):capture_initial_state(path,normal_target=tensor)
