So we want to figure out how to replicate the computation.

This has two implications:
1. replicate the forward computation
2. replicate the backward computation (you know, we are doing gradient based machine learning)

For the second case, we need some more work.

1. We know how gradient tape work when we define a custom op or (graph) function.
2. but that happens in a single computation replica.
3. How tape behaves in a replica context?
