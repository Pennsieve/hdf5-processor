import math
import os.path
import json
import numpy as np
import boto3
import h5py

from botocore.client import Config

from base_processor import BaseProcessor


class HDF5TypeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for HDF5 and numpy types.
    """
    def default(self, obj):
        if isinstance(obj, (np.int_,
                            np.intc,
                            np.intp,
                            np.int8,
                            np.int16,
                            np.int32,
                            np.int64,
                            np.uint8,
                            np.uint16,
                            np.uint32,
                            np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, h5py.Reference):
            return "<#Reference#>"
        if isinstance(obj, TreeNode):
            return obj.as_dict()
        else:
            try:
                return json.JSONEncoder.default(self, obj)
            except json.JSONDecodeError:
                return "<#Invalid#>"


class TreeNode(object):
    """
    TreeNode used to construct a tree for HDF5 export.
    """
    @classmethod
    def nullify_invalid_json_values(cls, v):
        """
        Transforms invalid JSON values into `None`. Valid values are returned
        unmodified.

        For floats:
            - NaN
            - +/-Inf

        Parameters
        ----------
        v : mixed

        Returns
        -------
        mixed
        """
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        else:
            return v

    @classmethod
    def key_value_list(cls, attribute_map):
        """
        Transform a map of key values, to a list consisting of
        { "key": <str>, "value": ... } object.

        Parameters
        ----------
        attribute_map : Dict[str, any]

        Returns
        -------
        List[{ "key": <str>, "value": ... }]
        """
        return [dict(key=key, value=cls.nullify_invalid_json_values(value))
                for (key, value) in attribute_map.items()]

    def __init__(self, name, path, hdf5_object, flattened=False):
        """
        Parameters
        ----------
        name : str
            TreeNode name
        path : List[str]
            The path of the TreeNode in the tree
        hdf5_object : h5py.*
            The HDF5 object to wrap
        flattened : bool
            If true, the TreeNode will be presented in a "flattened" form
        """
        self.name = name
        self.path = path
        self._hdf5_object = hdf5_object
        self._children = []
        self.flattened = flattened

    @property
    def path_key(self):
        return '/'.join(self.path)

    @property
    def parent_path_key(self):
        if len(self.path) <= 1:
            return '/'
        else:
            return '/'.join(self.path[:-1])

    @property
    def children(self):
        return self._children

    def add_child(self, child):
        self._children.append(child)
        return self

    def __len__(self):
        return len(self._children)

    def _get_shape(self, shape):
        """
        len(())         => 1          # scalar
        len((10000,))   => 10000      # 10000 element, 1D array
        len((500, 500)) => (500, 500) # 500 x 500 2D array
        return anything else
        """
        if not shape:
            return 1
        if len(shape) == 1:
            return shape[0]
        return shape

    def _get_dimensions(self, shape):
        if not shape or len(shape) <= 1:
            return 1
        else:
            return len(shape)

    def _is_scalar_object(self, hdf5_object):
        s = hdf5_object.shape
        return self._get_dimensions(s) == 1 and self._get_shape(s) == 1

    def as_dict(self):
        """
        Return the TreeNode as a dict of the form

            {
              "name": "<NAME>",
              "path": ["path1", "path2", ..., "pathN"],
              "path_key": "path1/path2/.../pathN",
              "type": {
                "type": "float32",
                "dimensions": 1,
                "size": 1000
              },
              "value": 99, # Only present if the type represents a scalar value
              "metadata: [
                {
                  "key": "key1",
                  "value": "value1"
                },
                ...
                {
                  "key": "keyN",
                  "value": "valueN"
                }
              ]
            }
        """
        if isinstance(self._hdf5_object, h5py.Dataset):
            shape = self._hdf5_object.shape
            type_info = dict(
                type=str(self._hdf5_object.dtype),
                dimensions=self._get_dimensions(shape),
                size=self._get_shape(shape)
            )
        else:
            type_info = "group"
        d = dict(name=self.name,
                 path=self.path,
                 path_key=self.path_key,
                 type=type_info,
                 metadata=self.key_value_list(self._hdf5_object.attrs))
        # Only output scalar values:
        if isinstance(self._hdf5_object, h5py.Dataset) and \
           self._is_scalar_object(self._hdf5_object):
            d["value"] = self.nullify_invalid_json_values(self._hdf5_object[()])
        if not self.flattened:
            d["children"] = self._children
        return d


def extract(file_name, flatten=False):
    """
    Given a HDF5 filename, extract the structure of the HDF5 file, returning
    the structure as a tree or flattened list.

    Parameters
    ----------
    file_name : str
        The HDF5 file to inspect
    flatten : bool
        If True, the returned struct will be a list of TreeNodes, otherwise a
        list of TreeNode instances without children will be returned.

    Returns
    -------
    TreeNode|List[TreeNode]
    """
    hdf5_file = h5py.File(file_name, 'r')

    linearized_nodes = []
    path_to_node = {}

    def walk(root, root_node, ancestors):
        for (name, hdf5_object) in root.items():
            if isinstance(hdf5_object, h5py.Dataset) or \
               isinstance(hdf5_object, h5py.Group):
                node = TreeNode(name=name,
                                path=ancestors + [name],
                                hdf5_object=hdf5_object,
                                flattened=flatten)
                if flatten:
                    # linear structure:
                    linearized_nodes.append(node)
                else:
                    # tree structure:
                    if node.path_key not in path_to_node:
                        path_to_node[node.path_key] = node
                    if node.parent_path_key in path_to_node:
                        path_to_node[node.parent_path_key].add_child(node)
                    if node.parent_path_key == "/":  # Parent is root
                        root_node.add_child(node)
                if hdf5_object and isinstance(hdf5_object, h5py.Group):
                    walk(hdf5_object, root_node, ancestors + [name])

    root_node = TreeNode(name="/", path=[], hdf5_object=hdf5_file,
                         flattened=flatten)
    walk(hdf5_file, root_node, [])

    return linearized_nodes if flatten else root_node


def extract_as_json(file_name, flatten=False):
    """
    Extract the structure of the HDF5 file as JSON.

    Parameters
    ----------
    file_name : str
        The HDF5 file to inspect
    flatten : bool
        If True, the returned struct will be a list of TreeNodes, otherwise a
        list of TreeNode instances without children will be returned.

    Returns
    -------
    str
    """
    return json.dumps(extract(file_name, flatten=flatten), cls=HDF5TypeEncoder)


class HDF5StructureProcessor(BaseProcessor):
    required_inputs = ["file"]
    node = TreeNode

    def __init__(self, *args, **kwargs):
        super(HDF5StructureProcessor, self).__init__(*args, **kwargs)
        self.session = boto3.session.Session()
        self.s3_client = self.session.client('s3', config=Config(signature_version='s3v4'), endpoint_url=self.settings.s3_endpoint)
        self.file = self.inputs.get('file')

        # structure JSON file:
        self.payload_upload_key = None
        self.payload_output_path = None

        # asset JSON file:
        self.asset_upload_key = None
        self.asset_output_path = None

    def get_file_size(self, key):
        self.LOGGER.info('Getting file size of {key}'.format(key=key))
        response = self.s3_client.head_object(
            Bucket=self.settings.storage_bucket, Key=key
        )
        file_size = response['ContentLength']
        self.LOGGER.info('{key}: size in bytes: {size}'.format(key=key,
                                                               size=file_size))
        return file_size

    def task(self):
        # write out the structure as JSON:
        output_file_name = "{job}-{name}.json".format(
                job=self.settings.aws_batch_job_id,
                name=os.path.basename(self.file))

        # build the upload key for the structure JSON file:
        self.payload_upload_key = os.path.join(self.settings.storage_directory,
                                               output_file_name)

        # write the local structure JSON file out:
        local_payload_dest = os.path.join(self.settings.scratch_dir,
                                          output_file_name)

        self.LOGGER.info('writing out local payload = {dest}'
                         .format(dest=local_payload_dest))

        with open(local_payload_dest, 'w') as f:
            self.LOGGER.info("Writing out {local_payload_dest}"
                             .format(local_payload_dest=local_payload_dest))
            f.write(extract_as_json(self.file, flatten=True))
            self.payload_output_path = local_payload_dest

        # upload the structure JSON file:
        self.LOGGER.info('upload payload = {key}'
                         .format(key=self.payload_upload_key))
        self._upload(self.payload_output_path, self.payload_upload_key)

        # get the file size of the structure JSON file:
        file_size = self.get_file_size(self.payload_upload_key)

        # build and write out the asset file:
        asset_info = {
            'bucket': self.settings.storage_bucket,
            'key': self.payload_upload_key,
            'type': 'view',
            'size': file_size
        }

        self.asset_output_path = "asset_info.json"
        self.publish_outputs("asset_info", asset_info)

        # upload the asset file:
        self.LOGGER.info('upload asset = {key}'
                         .format(key=self.asset_output_path))
        self._upload(local_payload_dest, self.asset_output_path)
