from colorama import init, Fore, Style
init()

import tensorflow as tf


data_parallel = tf.distribute.MirroredStrategy(devices=tf.config.list_logical_devices("GPU"))

@tf.function
def print_in_ctx(v):
    ctx = tf.distribute.get_replica_context()
    if ctx:
        # NOTE: print and tf.print is different!
        tf.print("Replica id:", ctx.replica_id_in_sync_group, "of", ctx.num_replicas_in_sync)
        # we should never use python's builtin print for strategy debug
        # print("Replica id: ", ctx.replica_id_in_sync_group," of ", ctx.num_replicas_in_sync)

        tf.print("Original Value:\n", v)

        v += float(ctx.replica_id_in_sync_group)
        tf.print("Modified Value:\n", v)

        print(dir(ctx))
    else:
        tf.print("ctx is None!\nValue:\n", v)

with data_parallel.scope():
    # v = tf.Variable(tf.random.normal((2,2), stddev=0.01))
    v = tf.Variable(tf.zeros((2,2)))

    # NOTE: distribute strategy scope only affect variable creation, execution is not affected.


    print(f"\n{Fore.GREEN}================ 1 ================{Style.RESET_ALL}")
    print(v)

    print(f"\n{Fore.GREEN}================ 2 ================{Style.RESET_ALL}")
    data_parallel.run(print, args=(v,))

    print(f"\n{Fore.GREEN}================ 3 ================{Style.RESET_ALL}")
    # Run `fn` on each replica, with the given arguments.
    # NOTE: strategy.run will affect execution
    data_parallel.run(print_in_ctx, args=(v,))

    print(f"\n{Fore.GREEN}================ 4 ================{Style.RESET_ALL}")
    # What if we run it without replica context? The value in replica 0 will be printed.
    # NOTE: the following 2 lines do the same thing.
    tf.print(v)
    print_in_ctx(v)
