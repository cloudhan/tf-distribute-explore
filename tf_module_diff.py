import types
import tensorflow as tf

v1 = tf.compat.v1
v2 = tf.compat.v2

def get_modules(top, no_private=True):
    mods = []
    for name in dir(top):
        if no_private and name.startswith("_"):
            continue
        member = getattr(top, name)
        if isinstance(member, types.ModuleType):
            mods.append(name)
    return mods

v1_mods = set(get_modules(v1))
v2_mods = set(get_modules(v2))

shared = v1_mods.intersection(v2_mods)
v1_only = sorted(list(v1_mods - shared))
v2_only = sorted(list(v2_mods - shared))
shared = sorted(list(shared))

print("v1 only (deprecated):")
print(v1_only)
print()

print("v2:")
print(shared)
print("v2 only:")
print(v2_only)
