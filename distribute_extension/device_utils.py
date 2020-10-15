import tensorflow as tf

_VIRTUAL_GPUS = None

def limit_to_virtual_gpus(num_vgpus=4):
    global _VIRTUAL_GPUS
    if _VIRTUAL_GPUS is not None:
        return _VIRTUAL_GPUS
    else:
        gpus = tf.config.list_physical_devices(device_type="GPU")
        assert len(gpus) > 0, "No GPU found."
        tf.config.set_visible_devices(gpus[0], device_type="GPU")
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024) for i in range(num_vgpus)]
        )
        gpus = tf.config.list_logical_devices("GPU")
        _VIRTUAL_GPUS = gpus
        return gpus


if __name__ == "__main__":
    gpus = limit_to_virtual_gpus(7)
    for i, g in enumerate(gpus):
        print(g.device_type, i)

    ms = tf.distribute.MirroredStrategy(devices=gpus)
