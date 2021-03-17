---
author: Philip Adams
title: Hierarchical PQ for ANN Search
description: Results from a series of experiments using 'prior' data to improve PQ recall
date: 
slug: multi-pq
tags: ["ANN Search", "Quantization","Systems", "C++"]
image: matryoshka-doll.jpg 
draft: true
---

# Background

Approximate Nearest Neighbors (ANN) Search is a simple problem: given a vector query vector $q$, a set of database vectors $V$, and a metric $d$, find the $v\in V$ that minimize $d(q, v)$. The *approximate* part of this problem is that an algorithm is allowed to be wrong, and so in particular doesn't need to test every $v\in V$, allowing for significant improvements in performance through the use of various search structures and compression techniques.

An active research community has sprung up around building new techniques for this problem, driven by its importance to semantic search, where documents are embedded into a vector space to perform the search. [Facebook](https://github.com/facebookresearch/faiss), [Google](https://github.com/google-research/google-research/tree/master/scann), [Microsoft](https://github.com/microsoft/SPTAG), and [Yahoo](https://github.com/yahoojapan/NGT) have all released productionized systems for ANN search, based on work done by research teams at each organization. There is even a site providing head-to-head benchmarks of different systems on reference datasets, at [ann-benchmarks.com](http://ann-benchmarks.com).

One well-known compression technique is Product Quantization (PQ), first applied to ANN search in 2010 by Jegou et al[^pqcite]. PQ is parameterized by two variables, $M$ and $nbits$. It splits each vector $v$ of dimension $d$ into $M$ subvectors $v_1,\dots, v_M$ of dimension $\frac{d}{M}$. Then, it trains $M$ codebooks with $2^{nbits}$ entries each using K-means clustering on the sets of subvectors. Now, every vector can be represented as a code of length $M\cdot nbits$. An additional benefit of a coding scheme is that rather than performing expensive distance calculations between vectors, a lookup table with $M\cdot 2^{nbits} \cdot 2^{nbits}$ can be constructed, and distance calculations are reduced to $M$ lookups in this table. Query vectors can be handled similarly, either by encoding the query vector under the PQ scheme and using the existing table (called Symmetric Distance Calculation (SDC)), or by computing a query-specific distance table of size $M \cdot nbits$ (called Asymmetric Distance Calculation (ADC)). For large datasets, these distance table approaches are significantly faster than normal distance calculations between unquantized vectors.

{{<figure src="/blog/img/pq_encode.gif" caption="Process of encoding a vector with PQ">}}

There have been a number of research attempts to improve on the performance of PQ. An early observation was that the standard PQ approach is reliant on the basis the data is presented in. For an extreme example, suppose that we are quantizing four-dimensional vectors with $M=2, nbits=1$, and the vectors are:

$$ V = \\left\\{ [1,0,1,0], [0,1,0,1], [1,1,1,1], [0,0,0,0] \\right\\}$$

In this case, we have that the first and third dimension are perfectly correlated, and similarly with the second and fourth, and can observe that if we were able to swap the second and third dimensions we would be able to quantize the vectors losslessly. Optimized PQ (OPQ), first presented by Ge et al. in 2013,[^opqcite] gives us a way to do that. Before training the codebooks for each subvector, we first train an orthogonal matrix, which is applied to all the vectors before they are quantized. This allows us 'waste' fewer bits quantizing redundant information.

Another attempt to improve the recall-per-bit performance of PQ is Composite Quantization (CQ), presented by Zhang et al. in 2014[^cqcite]. In CQ, the training algorithm is given even more degrees of freedom to optimize with, as rather than splitting the vector into subvectors, it only requires that the subspaces spanned by the codebooks be mutually near-orthogonal, and reconstructs the resultant vector from codes through summation rather than concatenation of subvectors. This is a direct generalization of PQ to a larger configuration space, and the main contribution is the derivation of an efficient training method for the codebooks.

One thing that all of these attempts to improve PQ have in common is that they all attempt to make improvements solely from the information contained in the data vectors themselves, rather than from information about the distribution of queries that will be made against that data. They mainly accomplish this goal by finding new ways to expand the optimization space the trainer can explore without significantly increasing the cost of training. Improvements that can be made purely by looking at the data are more desirable, since we can be more confident in their robustness to different applications and/or query distributions. Additionally, there is already a well understood literature and set of benchmarks for evaluating the performance of purely data-dependent ANN approaches. However, this focus on query-independent approaches may leave some low-hanging fruit for application that exhibit stable query distributions, such as web search.

In web search, the distribution of queries is known to be long-tailed. Because of this phenomenon, if we can expend extra resources on encoding/searching for vectors that are NNs for queries that occur frequently, we would expect to get an outsized return on those resources. Additionally, since those vectors make up a small portion of the dataset, the cost is not significantly higher than a conventional technique and so we do not expect degradation in tail performance.

# Definition of 'Prior' data and Hierarchical PQ
In the experiments discussed in this post, we consider two sets of queries, $Q_{\mathrm{train}}$ and $Q_{\mathrm{test}}$, drawn independently from the same distribution. We use the $Q_{\mathrm{train}}$ queries to compute the priors for each vector in the dataset. We define two types of prior, the linear and exponential falloff prior, as follows:

$$ \mathrm{NN}(q,v) = k : (v \text{ is a } k \text{-NN of } q \wedge \nexists k' < k : v \text{ is a } k' \text{-NN of } q)$$
$$ p_\mathrm{linear}(Q,v,k) = \frac{1}{|Q|} \cdot \sum_{q\in Q} (\mathrm{NN}(q,v) \leq k)$$
$$ p_\mathrm{expfalloff}(Q,v) = \frac{1}{|Q|} \cdot \sum_{q\in Q} e^{-\mathrm{NN}(q,v)}$$

Essentially, the more likely the vector is to come up as a result in the searches on $Q_{\mathrm{train}}$, the higher the prior. We then conduct the actual searches using $Q_{\mathrm{test}}$, which was drawn independently.

What is the best way to use this prior information? We had a number of ideas, including using the prior data during the construction of the search structure to make high-prior vectors more likely to be 'touched' by any given search[^hnswidea]. We opted for a more straightforward approach, though: using more bits to encode vectors that are high-prior than those that are low-prior (and thus encoding high-prior vectors more accurately). This is the main idea behind the technique explored in the rest of this post, which we call *Hierarchical PQ* (HPQ). In HPQ, rather than having one codebook per subvector, we have a hierarchy of codebooks $C^0 \subset C^1 \subset \dots \subset C^n$. We can decide which codebook to quantize any given vector with according to the prior information. Crucially, because the codebooks contain each other, they are *compatible*: we can use a distance table built on $C^n$, the codebook at the top of the hierarchy, to compute the distance between two vectors quantized by different levels on the hierarchy. This means that when handling queries, we only have to build the ADC distance table once per query. Because the distance table is so frequently accessed during search, this has the additional benefit of better cache performance from spatial locality versus an approach with multiple disjoint codebooks.

![HPQ Codebook](hpq_cb.png)
![HPQ Distance Table](hpq_dt.png)

# TODO Remove all the off-by-one errors in the diagrams

# Implementation Basics

- Current implementation only handles 2 codebooks
- K-means with fixed points?
- Codes/lookup table/change to storing which table inside code?

# Performance Characteristics

- Performance of decoding/masking non-8-bits
- Comparisons with 2-index approach (need to merge results, fixed costs)

# Results

- Recall@R, QPS on exact, HNSW

# Acknowledgments 
The majority of the work on this project was done as part of a course at UChicago, CMSC 33550, taught/advised by [Raul Castro Fernandez](https://raulcastrofernandez.com/). My partner for the project was my friend [Abdo](https://github.com/akabdo).

---
Note: the [featured image for this post](https://foto.wuestenigel.com/opening-matryoshka-dolls/) is by Marco Verch, and is licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).


[^pqcite]: Jegou, Herve, Matthijs Douze, and Cordelia Schmid. "Product quantization for nearest neighbor search." *IEEE transactions on pattern analysis and machine intelligence* 33, no. 1 (2010): 117-128.
[^opqcite]: Ge, Tiezheng, Kaiming He, Qifa Ke, and Jian Sun. "Optimized product quantization." *IEEE transactions on pattern analysis and machine intelligence* 36, no. 4 (2013): 744-755.
[^cqcite]: Zhang, Ting, Chao Du, and Jingdong Wang. "Composite quantization for approximate nearest neighbor search." In *International Conference on Machine Learning*, pp. 838-846. PMLR, 2014.
[^hnswidea]: We actually implemented a form of this idea for a search structure called Hierarchical Navigable Small Worlds (HNSW), first presented by Malkov et al.[^hnswcite] in 2018. HNSW works by creating multiple layers, and selecting which layer each vector should have as it's highest layer by sampling from an exponential distribution. So, the top layer will have very few vectors, and the lowest layer contains the whole dataset. Within each layer, a neighborhood graph is constructed, allowing for search within the layer. A full search starts at the top layer, searches within it, then goes down into the next layer, searches within that layer, and so on until the bottom layer is reached. This technique achieves some of the best results on [ann-benchmarks.com](http://ann-benchmarks.com). We implemented a modification where instead of selecting a vector's layer by sampling from a random distribution, it selected based on the sample plus the prior. This did have the effect of pushing high-prior vectors to higher layers, meaning that they were seen by more searches. However, in our experiments, that structural change did not have any impact on performance. Investigating this further may be the topic of a future blog post!
[^hnswcite]: Malkov, Yu A., and Dmitry A. Yashunin. "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs." *IEEE transactions on pattern analysis and machine intelligence* 42, no. 4 (2018): 824-836.
