"""The declared texture raster contract matches effective source policy."""
import json
from pathlib import Path

import pytest
from utils.texture_raster_budget import source_raster_policy


def test_source_texture_raster_capability_declaration():
    path = Path(__file__).parents[1] / 'seed_backend_capabilities.json'
    data = json.loads(path.read_text())
    assert data['contract'] == 'seed.pshuman.backend_capabilities.v1'
    assert data['features']['source_texture_raster'] == {
        'version': 1,
        'frozenAcrossViewsAndPasses': True,
        'binCapacityMode': 'all_faces_or_default',
        'bufferLimitBytes': 'enforced_for_all_faces',
    }
    assert source_raster_policy({}) == {'mode': 'default', 'bin_buffer_limit_bytes': None}
    assert source_raster_policy({'PSHUMAN_TEXTURE_RASTER_MODE':'all_faces'}) == {
        'mode':'all_faces','bin_buffer_limit_bytes':536870912}
    assert source_raster_policy({'PSHUMAN_TEXTURE_RASTER_MODE':'all_faces',
                                 'PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES':'123'}) == {
        'mode':'all_faces','bin_buffer_limit_bytes':123}
    with pytest.raises(ValueError):
        source_raster_policy({'PSHUMAN_TEXTURE_RASTER_MODE':'all_faces',
                              'PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES':'0'})
