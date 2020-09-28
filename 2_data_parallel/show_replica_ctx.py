from colorama import init, Fore, Style
init()

import tensorflow as tf


data_parallel = tf.distribute.MirroredStrategy(devices=tf.config.list_logical_devices("GPU"))

@tf.function
def fn():
    ctx = tf.distribute.get_replica_context()
    tf.print(ctx)
    if ctx:
        tf.print("Replica id:", ctx.replica_id_in_sync_group, "of", ctx.num_replicas_in_sync)
    else:
        tf.print("ctx is None!")


print(f"\n{Fore.GREEN}================ 1 ================{Style.RESET_ALL}")
fn()

with data_parallel.scope():
    print(f"\n{Fore.GREEN}================ 2 ================{Style.RESET_ALL}")
    fn()

    print(f"\n{Fore.GREEN}================ 3 ================{Style.RESET_ALL}")
    data_parallel.run(fn)


print(f"\n{Fore.GREEN}================ concrete example ================{Style.RESET_ALL}")
@tf.function
def create():
    """We generally don't do this, simply do:
    ```
    with strategy.scope():
        v = tf.Variable(...)
    ```
    """
    ctx = tf.distribute.get_replica_context()
    assert ctx is not None, "this function should be called in replica ctx"
    # ctx.replica_id_in_sync_group is a tensor!
    return tf.cast(ctx.replica_id_in_sync_group, dtype=tf.float32)

@tf.function
def sum(dist_v):
    ctx = tf.distribute.get_replica_context()
    assert ctx is None, "this function should be called in cross-replica ctx"
    strategy = tf.distribute.get_strategy()

    # obtain scalar with None axis
    return strategy.reduce(tf.distribute.ReduceOp.SUM, dist_v, axis=None)

with data_parallel.scope():
    dist_v = data_parallel.run(create)
    tf.print("distribute value:", dist_v)
    tf.print("reduced value:", sum(dist_v))
