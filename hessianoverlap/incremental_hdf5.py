#!/usr/bin python
# -*- coding:utf-8 -*-


"""
HDF5 is a serialization format that allows to efficiently access array-like
data from persistent memory (e.g. disk), as if it was volatile (e.g. RAM).
It is useful when we need to handle large amounts of numerical data (e.g.
NumPy arrays) and/or when we need it to persist across sessions/devices.

This module implements convenience interfaces to store and access HDF5 data.
"""


import os

#
import numpy as np

import h5py


# ##############################################################################
# # I/O
# ##############################################################################
class IncrementalHDF5:
    """
    Incrementally concatenate matrices of same height. Note the usage of very
    few datasets, to prevent slow loading times. Usage example::


      # GATHERING SESSION:
      h5 = IncrementalHDF5(path, height, dtype=np.float32,
                           compression="lzf", err_if_exists=True)
      for step in range(100):
          # Append some random data and metadata
          h5.append(np.random.randn(height, 5), str({"step": step}))
      h5.close()

      # RETRIEVING SESSION:
      h5 = h5py.File(path, "r")
      for i in range(IncrementalHDF5.get_num_elements(h5)):
            data, meta = IncrementalHDF5.get_element(h5, i)
            meta = ast.literal_eval(meta)
            print("Range and step:", (data.min(), data.max()), meta["step"])
    """

    DATA_NAME = "data"
    METADATA_NAME = "metadata"
    IDXS_NAME = "data_idxs"

    def __init__(
        self,
        out_path,
        height,
        dtype=np.float32,
        compression="lzf",
        data_chunk_length=500,
        metadata_chunk_length=500,
        err_if_exists=True,
    ):
        """
        :param height: This class incrementally stores a matrix of shape
          ``(height, w++)``, where ``height`` is always fixed.
        :param compression: ``lzf`` is fast, ``gzip`` slower but provides
          better compression
        :param data_chunk_length: Every I/O operation goes by chunks. A too
          small chunk size will cause many syscalls (slow), and with a too large
          chunk size we will be loading too much information in a single
          syscall (also slow, and bloats the RAM). Ideally, the chunk length is
          a bit larger than what is usually needed (e.g. if we expect to read
          between 10 and 50 rows at a time, we can choose chunk=60).
        """
        self.out_path = out_path
        self.height = height
        self.dtype = dtype
        self.compression = compression
        #
        if err_if_exists:
            if os.path.isfile(out_path):
                raise FileExistsError(f"File already exists! {out_path}")
        #
        self.h5f = h5py.File(out_path, "w")
        self.data_ds = self.h5f.create_dataset(
            self.DATA_NAME,
            shape=(height, 0),
            maxshape=(height, None),
            dtype=dtype,
            compression=compression,
            chunks=(height, data_chunk_length),
        )
        self.metadata_ds = self.h5f.create_dataset(
            self.METADATA_NAME,
            shape=(0,),
            maxshape=(None,),
            compression=compression,
            dtype=h5py.string_dtype(),
            chunks=(metadata_chunk_length,),
        )
        self.data_idxs_ds = self.h5f.create_dataset(
            self.IDXS_NAME,
            shape=(2, 0),
            maxshape=(2, None),
            dtype=np.int64,
            compression=compression,
            chunks=(2, metadata_chunk_length),
        )
        self._current_data_width = 0
        self._num_entries = 0

    def __enter__(self):
        """ """
        return self

    def __exit__(self, type, value, traceback):
        """ """
        self.close()

    def close(self):
        """ """
        self.h5f.close()

    def append(self, matrix, metadata_str):
        """
        :param matrix: dtype array of shape ``(fix_height, width)``
        """
        n = self._num_entries
        h, w = matrix.shape
        assert (
            h == self.height
        ), f"Shape was {(h, w)} but should be ({self.height}, ...). "
        # update arr size and add data
        new_data_w = self._current_data_width + w
        self.data_ds.resize((self.height, new_data_w))
        self.data_ds[:, self._current_data_width : new_data_w] = matrix
        # # update meta-arr size and add metadata
        self.metadata_ds.resize((n + 1,))
        self.metadata_ds[n] = metadata_str
        # update data-idx size and add entry
        self.data_idxs_ds.resize((2, n + 1))
        self.data_idxs_ds[:, n] = (self._current_data_width, new_data_w)
        #
        self.h5f.flush()
        self._current_data_width = new_data_w
        self._num_entries += 1

    @classmethod
    def get_element(cls, h5file, elt_idx, with_data=True, with_metadata=True):
        """
        :param int elt_idx: Index of the appended element, e.g. first element
          has index 0, second has index 1...
        :returns: the ``(data, metadata_str)`` corresponding to that index,
          as they were appended.
        """
        data_beg, data_end = h5file[cls.IDXS_NAME][:, elt_idx]
        data, metadata = None, None
        if with_data:
            data = h5file[cls.DATA_NAME][:, data_beg:data_end]
        if with_metadata:
            metadata = h5file[cls.METADATA_NAME][elt_idx].decode("utf-8")
        return data, metadata

    @classmethod
    def get_num_elements(cls, h5file):
        """
        :returns: The number of elements that have been added to the file via
          append.
        """
        num_elements = len(h5file[cls.METADATA_NAME])
        return num_elements
