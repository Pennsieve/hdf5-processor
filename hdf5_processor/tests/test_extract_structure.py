#!/usr/bin/env python

import os.path
import json

from moto import mock_s3
from moto import mock_ssm
import pytest
from six import string_types

from base_processor.tests import init_ssm
from base_processor.tests import setup_processor

from hdf5_processor import HDF5StructureProcessor


test_processor_data = [
    'H19.29.141.11.21.01.nwb',
    'test.hdf5',
    'test.nwb'
]


def test_strip_nan_and_inf():
    sample_dict = {
        "nan_value": float('nan'),
        "inf_value": float('inf'),
        "normal_value": 100.123
    }
    key_value_list = HDF5StructureProcessor.node.key_value_list(sample_dict)
    assert len(key_value_list) == 3
    for kv in key_value_list:
        assert 'key' in kv and 'value' in kv
        if kv['key'] == 'nan_value':
            assert kv['value'] is None
        elif kv['key'] == 'inf_value':
            assert kv['value'] is None
        elif kv['key'] == 'normal_value':
            assert kv['value'] == 100.123


@pytest.mark.parametrize("filename", test_processor_data)
def test_extract_structure_as_json(filename):
    inputs = {'file': os.path.join('/test-resources', filename)}

    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    task = HDF5StructureProcessor(inputs=inputs)
    setup_processor(task)
    task.run()

    # Verify the payload JSON is written out:
    assert task.payload_output_path and os.path.exists(task.payload_output_path)
    with open(task.payload_output_path, 'r') as f:
        payload = json.load(f)
        assert len(payload) > 0
        entry = payload[0]
        assert 'path' in entry and isinstance(entry['path'], list)
        assert 'path_key' in entry and isinstance(entry['path_key'], string_types)
        assert 'name' in entry and isinstance(entry['name'], string_types)
        assert 'metadata' in entry and isinstance(entry['metadata'], list)
        assert 'type' in entry and isinstance(entry['type'], (dict, string_types))
        if isinstance(entry['type'], dict) and \
           entry['type']['dimensions'] == 1 and \
           entry['type']['size'] == 1:
            assert 'value' in entry

    # Verify the asset JSON is written out:
    assert task.asset_output_path and os.path.exists(task.asset_output_path)
    with open(task.asset_output_path, 'r') as f:
        asset_info = json.load(f)
        print(asset_info)
