"""
Read-only cache files without memory-mapping.

Trainers read large uint8 picture caches (e.g., work/eval/chest/taix_512.u8: 56.5 GB)
through np.memmap, and every page touched stays in the trainer's working set,
starving the PC. FileRows reads the same cache through ordinary file reads instead.
The OS file cache still caches pages, but they land on the standby list, not in the
trainer's working set. Idea 53's bench (tools/rows_bench.py) measures the working
set and the throughput against memmap.
"""

import numpy as np
import threading
import os


class FileRows:
    """Drop-in replacement for np.memmap(path, dtype, mode='r', shape=shape)."""

    def __init__(self, path, shape, dtype=np.uint8, offset=0):
        """
        Args:
            path: file path
            shape: (rows, *inner_shape) tuple; rows is the index dimension
            dtype: numpy dtype (default uint8)
            offset: byte offset to data start (default 0)
        """
        self.path = path
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.offset = offset
        self._row_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        self._handles = threading.local()

    @classmethod
    def npy(cls, path):
        """FileRows over a .npy file: shape, dtype and data offset come from its header."""
        with open(path, 'rb') as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f"{path}: unsupported .npy version {version}")
            offset = f.tell()
        if fortran:
            raise ValueError(f"{path}: fortran_order .npy is not supported")
        if dtype.hasobject:
            raise ValueError(f"{path}: object dtype .npy is not supported")
        return cls(path, shape, dtype=dtype, offset=offset)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def _get_handle(self):
        """Get or create a file handle for the current thread."""
        if not hasattr(self._handles, 'f'):
            self._handles.f = open(self.path, 'rb', buffering=0)
        return self._handles.f

    def close(self):
        """Close any open file handles."""
        if hasattr(self._handles, 'f'):
            self._handles.f.close()
            delattr(self._handles, 'f')

    def _normalize_index(self, idx):
        """Convert negative and out-of-range indices; raise IndexError if out of bounds."""
        if idx < 0:
            idx = self.shape[0] + idx
        if idx < 0 or idx >= self.shape[0]:
            raise IndexError(f"row index {idx} out of range for {self.shape[0]} rows")
        return idx

    def _read_rows(self, row_indices):
        """
        Read rows in the given order.

        Args:
            row_indices: 1-D array or list of row indices (may have duplicates, negatives)

        Returns:
            ndarray of shape (len(row_indices),) + self.shape[1:], dtype self.dtype
        """
        if isinstance(row_indices, list):
            row_indices = np.array(row_indices, dtype=np.int64)
        elif not isinstance(row_indices, np.ndarray):
            row_indices = np.asarray(row_indices, dtype=np.int64)

        # Normalize negative indices
        row_indices = np.where(row_indices < 0, self.shape[0] + row_indices, row_indices)

        # Check bounds
        if np.any(row_indices < 0) or np.any(row_indices >= self.shape[0]):
            bad = row_indices[(row_indices < 0) | (row_indices >= self.shape[0])][0]
            raise IndexError(f"row index {bad} out of range for {self.shape[0]} rows")

        # Find unique rows and their positions
        unique_rows, inverse_indices = np.unique(row_indices, return_inverse=True)

        # Group unique rows into consecutive runs
        runs = []
        if len(unique_rows) > 0:
            breaks = np.flatnonzero(np.diff(unique_rows) != 1)
            starts = np.concatenate(([0], breaks + 1))
            ends = np.concatenate((breaks, [len(unique_rows) - 1]))
            runs = [(int(unique_rows[s]), int(unique_rows[e])) for s, e in zip(starts, ends)]

        # Read all rows and build output
        uniq = np.empty((len(unique_rows),) + self.shape[1:], dtype=self.dtype)

        f = self._get_handle()
        pos = 0

        for start, end in runs:
            n_rows = int(end - start) + 1
            run_buffer = uniq[pos:pos + n_rows]
            pos += n_rows

            byte_offset = self.offset + start * self._row_bytes
            n_bytes = n_rows * self._row_bytes

            f.seek(byte_offset)
            view = memoryview(run_buffer).cast('B')
            got = 0
            while got < n_bytes:
                n_read = f.readinto(view[got:])
                if not n_read:
                    raise IOError(
                        f"end of file after {got} of {n_bytes} bytes "
                        f"at offset {byte_offset}"
                    )
                got += n_read

        output = uniq[inverse_indices.reshape(-1)]

        return output

    def __getitem__(self, key):
        """
        Support int, slice, list, ndarray, and tuple indexing.

        Returns fresh writable C-contiguous ndarray (never a memmap).
        """
        # Handle tuple: (first_element, *rest_are_slice_None)
        if isinstance(key, tuple):
            if len(key) == 0:
                raise TypeError("empty tuple")
            first = key[0]
            rest = key[1:]
            if not all(isinstance(k, slice) and k == slice(None) for k in rest):
                raise TypeError(
                    f"tuple indices after the first must be slice(None), got {key}"
                )
            key = first

        # Handle int
        if isinstance(key, (int, np.integer)):
            idx = self._normalize_index(int(key))
            rows = self._read_rows([idx])
            return rows[0]

        # Handle slice
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                # Convert step != 1 slice to an index array
                indices = np.arange(start, stop, step, dtype=np.int64)
                return self._read_rows(indices)
            else:
                # step == 1 slice
                if start >= stop:
                    return np.zeros((0,) + self.shape[1:], dtype=self.dtype)
                indices = np.arange(start, stop, dtype=np.int64)
                return self._read_rows(indices)

        # Handle list or ndarray
        if isinstance(key, (list, np.ndarray)):
            if isinstance(key, np.ndarray) and key.dtype == np.bool_:
                raise TypeError(
                    f"boolean indexing is not supported; use integer array instead"
                )
            return self._read_rows(key)

        # Anything else is unsupported
        raise TypeError(
            f"unsupported index type: {type(key).__name__}. "
            f"Use int, slice, list, ndarray, or tuple with slice(None) tail."
        )


def selftest():
    """
    Validate FileRows against np.memmap.

    Creates a temp file, tests various indexing modes, threading, and edge cases.
    Prints 'rows selftest ok' on success.
    """
    import tempfile
    import zlib

    # Create temp file: 100 rows x 6 x 5 uint8
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, (100, 6, 5), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(data.tobytes())

    try:
        memmap = np.memmap(tmp_path, dtype=np.uint8, mode='r', shape=(100, 6, 5))
        filerows = FileRows(tmp_path, (100, 6, 5), dtype=np.uint8, offset=0)

        # Test: single int index
        assert np.array_equal(filerows[0], memmap[0])
        assert np.array_equal(filerows[50], memmap[50])
        assert np.array_equal(filerows[-1], memmap[-1])

        # Test: slice
        assert np.array_equal(filerows[3:17], memmap[3:17])
        assert np.array_equal(filerows[:5], memmap[:5])
        assert np.array_equal(filerows[95:], memmap[95:])
        assert np.array_equal(filerows[-10:-2], memmap[-10:-2])

        # Test: slice with step != 1
        assert np.array_equal(filerows[::3], memmap[::3])
        assert np.array_equal(filerows[1:20:2], memmap[1:20:2])

        # Test: list
        indices = [10, 20, 5, 20, 99]
        assert np.array_equal(filerows[indices], memmap[indices])

        # Test: ndarray fancy indexing
        idx_array = np.array([0, 50, 99, 50], dtype=np.int64)
        assert np.array_equal(filerows[idx_array], memmap[idx_array])

        # Test: empty list
        assert filerows[[]] .shape == (0, 6, 5)
        assert np.array_equal(filerows[[]], memmap[[]])

        # Test: tuple with trailing slice(None)
        assert np.array_equal(filerows[(10, slice(None), slice(None))], memmap[10])
        assert np.array_equal(filerows[([5, 10, 20], slice(None), slice(None))],
                              memmap[[5, 10, 20]])

        # Test: uint16 file with offset
        data16 = rng.integers(0, 65536, (50, 8, 8), dtype=np.uint16)
        header = b'\x00' * 128
        with tempfile.NamedTemporaryFile(delete=False) as tmp16:
            tmp16_path = tmp16.name
            tmp16.write(header)
            tmp16.write(data16.tobytes())

        try:
            memmap16 = np.memmap(tmp16_path, dtype=np.uint16, mode='r',
                                  offset=128, shape=(50, 8, 8))
            filerows16 = FileRows(tmp16_path, (50, 8, 8), dtype=np.uint16, offset=128)
            assert np.array_equal(filerows16[10], memmap16[10])
            assert np.array_equal(filerows16[5:15], memmap16[5:15])
            filerows16.close()
            del memmap16
        finally:
            try:
                os.unlink(tmp16_path)
            except PermissionError:
                pass

        # Test: out of range raises IndexError
        try:
            filerows[100]
            raise AssertionError("should have raised IndexError")
        except IndexError:
            pass

        try:
            filerows[[0, 50, 150]]
            raise AssertionError("should have raised IndexError")
        except IndexError:
            pass

        # Test: bool mask raises TypeError
        try:
            filerows[np.array([True, False] * 50)]
            raise AssertionError("should have raised TypeError")
        except TypeError:
            pass

        # Test: threading
        results = {}
        def read_thread(idx):
            results[idx] = filerows[idx].copy()

        threads = []
        for i in range(4):
            t = threading.Thread(target=read_thread, args=(i * 25,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        for i in range(4):
            assert np.array_equal(results[i * 25], memmap[i * 25])

        filerows.close()
        del memmap

    finally:
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass

    # FileRows.npy against np.load(mmap_mode='r')
    for arr in (rng.integers(0, 256, (40, 7, 9), dtype=np.uint8),
                rng.random((30, 5, 5), dtype=np.float32)):
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmpn:
            npy_path = tmpn.name
        try:
            np.save(npy_path, arr)
            mm = np.load(npy_path, mmap_mode='r')
            fr = FileRows.npy(npy_path)
            assert fr.shape == mm.shape and fr.dtype == mm.dtype
            n = arr.shape[0]
            assert np.array_equal(fr[3], mm[3])
            assert np.array_equal(fr[-1], mm[-1])
            assert np.array_equal(fr[2:n - 4], mm[2:n - 4])
            ix = np.array([n - 1, 0, 7, 7, 3], dtype=np.int64)
            assert np.array_equal(fr[ix], mm[ix])
            fr.close()
            del mm
        finally:
            try:
                os.unlink(npy_path)
            except PermissionError:
                pass
    for bad in (np.asfortranarray(np.zeros((4, 3, 2), np.uint8)),
                np.array([[1], [2]], dtype=object)):
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmpn:
            npy_path = tmpn.name
        try:
            np.save(npy_path, bad, allow_pickle=True)
            try:
                FileRows.npy(npy_path)
                raise AssertionError("should have raised ValueError")
            except ValueError:
                pass
        finally:
            try:
                os.unlink(npy_path)
            except PermissionError:
                pass

    print('rows selftest ok')
    return 0


if __name__ == '__main__':
    exit(selftest())
