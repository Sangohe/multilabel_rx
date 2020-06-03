import os
import sys
import glob
import importlib

import config


# ------------------------------------------------------------------------------------------------------
# Utilities for importing modules and objects by name.


def import_module(module_or_obj_name):
    parts = module_or_obj_name.split(".")
    parts[0] = {"np": "numpy", "tf": "tensorflow"}.get(parts[0], parts[0])
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            relative_obj_name = ".".join(parts[i:])
            return module, relative_obj_name
        except ImportError:
            pass
    raise ImportError(module_or_obj_name)


def find_obj_in_module(module, relative_obj_name):
    obj = module
    for part in relative_obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def import_obj(obj_name):
    module, relative_obj_name = import_module(obj_name)
    return find_obj_in_module(module, relative_obj_name)


def call_func_by_name(*args, func=None, **kwargs):
    assert func is not None
    return import_obj(func)(*args, **kwargs)


# ------------------------------------------------------------------------------------------------------
# Create result subdirectory


def create_result_subdir(result_dir, desc):
    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, "*")):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[: fbase.find("-")])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, "%03d-%s" % (run_id, desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print("Saving results to", result_subdir)
    set_output_log_file(os.path.join(result_subdir, "log.txt"))

    # Export config.
    try:
        with open(os.path.join(result_subdir, "config.txt"), "wt") as fout:
            for k, v in sorted(config.__dict__.items()):
                if not k.startswith("_"):
                    fout.write("%s = %s\n" % (k, str(v)))
    except:
        pass

    return result_subdir


# ------------------------------------------------------------------------------------------------------
# Logging of stdout and stderr to a file.


class OutputLogger(object):
    def __init__(self):
        self.file = None
        self.buffer = ""

    def set_log_file(self, filename, mode="wt"):
        assert self.file is None
        self.file = open(filename, mode)
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self):
        if self.file is not None:
            self.file.flush()


class TeeOutputStream(object):
    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush

    def write(self, data):
        for stream in self.child_streams:
            stream.write(data)
        if self.autoflush:
            self.flush()

    def flush(self):
        for stream in self.child_streams:
            stream.flush()


output_logger = None


def init_output_logging():
    global output_logger
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], autoflush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger], autoflush=True)


def set_output_log_file(filename, mode="wt"):
    if output_logger is not None:
        output_logger.set_log_file(filename, mode)
