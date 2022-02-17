---
author: Philip Adams
title: Some bounds for k-dle
description: 
date: 2022-02-16
slug: k-dle-bounds
tags: ["Optimization", "Wordle", "Math", "Fun", "Off-topic"]
image: k-dle.png
draft: true
toc: false
---
# Introduction
[Wordle](https://www.nytimes.com/games/wordle/index.html), the quintagram-finding game we all now know and love, has spawned countless variations, some [reasonable](https://nerdlegame.com/), some [fiendish](https://qntm.org/files/absurdle/absurdle.html), and some [hilarious](https://rsk0315.github.io/playground/passwordle.html). A variation I particularly enjoy, however, is $k$-dle, most notably [Dordle](https://zaratustra.itch.io/dordle) and [Quordle](https://www.quordle.com/). In these variants, the player is asked to guess multiple words at once, with each guess giving clues about each of the $k$ unique boards.

In an attempt to ride the bandwagon of [cool theoretical wordle results](https://www.poirrier.ca/notes/wordle-optimal/), and to find a way to beat the people close to me at Quordle, in this post I will share some bounds on $k$-dle strategies. I extend my apologies for any deficits in the quality of analysis or communication. 

# Model, Assumptions, and Basic Results

Luckily, $k$-dle is fairly easy to model. We can view a $k$-dle problem as a wordle problem with $5k$ letters, and for wordle guess and solution sets $G$ and $S$ respectively, $k$-dle sets $G_k = \\left\\{w^k \mid w \in G\\right\\}$, $S_k = S^k$. 

However, this allows multiple boards to have the same answer. It also allows for a lot of repetition by ordering the same boards differently. To avoid both these issues, we can issue the constraint 
$$\begin{align*} w\in S_k \iff &(\forall i \in [k], w_{k\cdot(i-1) +1}\dots w_{k\cdot(i-1) +5}\in S) \\\\  \wedge & (\forall i \in [k]\setminus \\{1\\}, j \in [i]\setminus \\{i\\},  w_{k\cdot(j-1) +1}\dots w_{k\cdot(j-1) +5} < w_{k\cdot(i-1) +1}\dots w_{k\cdot(i-1) +5} ).\end{align*}$$
Or, in plain English, a composite $k$-word is in the $k$-dle solution set if each subword is in the wordle solution set and all the subwords of the composite word are in increasing lexicographic order. Note that this lexicographic ordering constraint is a convenience for modeling, not something we can actually use in our strategies. That is, if ARISE is the word in the fourth word of a $4$-dle, we may not assume that the other three words also start with A in our strategies.  

Finally, we say that a $k$-dle has been solved if each subproblem has been solved by some guess.

With these formalities out of the way, we can start knocking out some easy results. Let's go!

## Result #1: $k$-dle requires at least $k$ guesses

This one is pretty easy. Our lexicographic ordering constraint implies that each subword is unique, so in the best case, where every guess gets one word correct, we will need to have as many guesses as subwords, $k$.

## Result #2: $k$-dle requires at most $|S|$ guesses

Again, a simple result. Exhausting the solution set will solve every $k$-dle puzzle. When $k=|S|$, this bound is tight.

## Result #3: $k$-dle requires at most $5k$ guesses

Here, we have a slightly more complicated result. We can consider solving each subproblem independent, with no shared information. This is just solving $k$ independent wordles. There is already an optimality result for wordle that shows it can be solved in 5 guesses at worst, so solving $k$ wordles independently will take at most $5k$ guesses. Two more general versions of this follow.

## Result #4: $\forall m,n\in \mathbb{N} \mid m+n=k$, $O(k\text{-dle}) \leq O(m\text{-dle}) + O(n\text{-dle})$

That is, $k$-dle can only be at most as hard as any way of splitting it up and solving those parts independently. This is clear from the same logic as the previous result.

## Result #5: $\forall m,n\in \mathbb{N} \mid m+n=k$, $E(k\text{-dle}) \leq E(m\text{-dle}) + E(n\text{-dle})$

Similarly, the average number of guesses to solve a $k$-dle is bounded above by the sum of the average number of guesses to solve any subpuzzles.

# Information-theoretic analysis

Grant Sanderson of the excellent Youtube channel/blog [3Blue1Brown](3b1b.co) made two [excellent]() [videos]() about using an information-theoretic heuristic for finding good strategies for solving wordle. I made some [modifications to his script]() to allow computing good first guesses and finding expected number of guesses for solving $k$-dles. I've computed these for $2$-dle through $8$-dle:

# Decision-tree analysis

# Plotting the Bounds

# Applying some OEISy-hot to our mental bruises
