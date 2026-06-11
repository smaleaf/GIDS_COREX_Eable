import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
from GIDS_IX import GIDS

print("=== GIDS-IX Quick Test ===")

page_size = 4096
cache_dim = 256
num_ele = 10000
num_ssd = 1
cache_size = 10

file_paths = ["/tmp/__gids_test_feat.bin"]

with open(file_paths[0], "wb") as f:
    data = b'\x00' * (num_ele * 4)
    f.write(data)

try:
    gids = GIDS(
        page_size=page_size,
        off=0,
        cache_dim=cache_dim,
        num_ele=num_ele,
        num_ssd=num_ssd,
        cache_size=cache_size,
        file_paths=file_paths,
        accumulator_flag=True,
        window_buffer=True,
        wb_size=8,
    )
    print("OK: GIDS instance created")
    print("  - page_size:", gids.page_size)
    print("  - file_paths:", gids.file_paths)
    print("  - cache_size:", gids.cache_size)
    print("  - accumulator_flag:", gids.accumulator_flag)
    print("  - window_buffering_flag:", gids.window_buffering_flag)

    print("OK: GIDS-IX is working correctly")
except Exception as e:
    print("ERROR:", e)
    import traceback
    traceback.print_exc()
finally:
    os.unlink(file_paths[0])