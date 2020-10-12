So we want to figure out how to replicate the computation.

This has two implications:
1. replicate the forward computation
2. replicate the backward computation (you know, we are doing gradient based machine learning)

For the first case, you simply: `strategy.run(computation_func)` as shown in data parallel example.
