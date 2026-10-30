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
