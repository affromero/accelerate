# Copyright 2026 Andres Romero. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import ctypes
import ntpath
import os
import re
import stat
from pathlib import Path


def validate_checkpoint_name(filename):
    if (
        not isinstance(filename, str)
        or not filename
        or "\\" in filename
        or "\x00" in filename
        or ntpath.splitdrive(filename)[0]
        or Path(filename).is_absolute()
        or ".." in Path(filename).parts
    ):
        raise ValueError("Checkpoint filenames must stay inside the checkpoint directory")


def _validate_target(root, target):
    if target.is_relative_to(root):
        return
    for snapshot in (root, *root.parents):
        if snapshot.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
            continue
        blobs = snapshot.parent.parent / "blobs"
        if blobs.is_symlink() or blobs.resolve(strict=True) != blobs:
            break
        if target.is_relative_to(blobs):
            return
    raise ValueError("Checkpoint file resolves outside the permitted checkpoint directory")


@contextlib.contextmanager
def open_checkpoint_file(checkpoint_folder, filename):
    validate_checkpoint_name(filename)
    root = Path(checkpoint_folder).resolve(strict=True)
    target = (root / filename).resolve(strict=True)
    _validate_target(root, target)
    with contextlib.ExitStack() as stack:
        if os.name == "posix":
            directory = os.open(target.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, directory)
            for component in target.parts[1:-1]:
                directory = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                stack.callback(os.close, directory)
            descriptor = os.open(target.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        else:
            descriptor = os.open(target, os.O_RDONLY | os.O_BINARY)
        stack.callback(os.close, descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Checkpoint file must be a regular file")
        if os.name == "nt":
            import msvcrt

            get_path = ctypes.windll.kernel32.GetFinalPathNameByHandleW
            get_path.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
            get_path.restype = ctypes.c_uint32
            buffer = ctypes.create_unicode_buffer(32768)
            length = get_path(msvcrt.get_osfhandle(descriptor), buffer, len(buffer), 0)
            if not length or length >= len(buffer):
                raise OSError("Cannot verify the opened checkpoint path")
            opened = buffer.value
            if opened.startswith("\\\\?\\UNC\\"):
                opened = "\\\\" + opened[8:]
            elif opened.startswith("\\\\?\\"):
                opened = opened[4:]
            if os.path.normcase(opened) != os.path.normcase(str(target)):
                raise ValueError("Checkpoint file changed while opening")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            yield handle
