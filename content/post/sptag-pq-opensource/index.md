---
author: Philip Adams
title: Product Quantization support in SPTAG is now Open Source
description: 
date: 2021-12-08
slug: sptag-pq
tags: ["ANN Search", "Quantization","Systems", "C++", "Open Source"]
image: 
draft: false
---

If you're reading this blog because you're interested in ANN search, this is a small update for you! The product quantization (PQ) support I've been working on in the [SPTAG project](https://github.com/microsoft/SPTAG) has been open-sourced. 

SPTAG is an ANN search algorithm that leverages a combination of a space partitioning tree (usually, a balanced k-means tree or k-d tree) and graph (usually, a relative neighborhood graph or KNN graph) to try and provide a good balance between high recall, reasonable index build times, and throughput. Product Quantization support offers a new trade-off between recall, memory usage, and throughput that may be a better fit for some scenarios. I discuss PQ in some more detail in [another post](/p/multi-pq) on this blog. 

There has been a lot of exciting open source development in SPTAG recently. Another feature (not my work) that was recently open-sourced is support for hybrid memory-disk indexes. This helps to serve very large indexes efficiently. You can read more about the hybrid index feature in [this paper](https://arxiv.org/pdf/2111.08566.pdf).

I plan on posting some benchmarks on a public dataset soon. Until then, if you're interested in the feature, [try it out!](https://github.com/microsoft/SPTAG/blob/main/docs/GettingStart.md)
