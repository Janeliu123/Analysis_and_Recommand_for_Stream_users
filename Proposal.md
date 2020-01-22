## Overview

Steam is an online game platform and the Steam community network is a large social network of players on this platform. Based on Steam, we want to perform online social network analysis with the following features:

1. Friends Recommendation

2. Games Recommendation

Supplementary:

1. Social networks graph visualization 

2. Other statistical data (similar to existing demos)

Steam has an average of 18.5 million concurrent players and 9300 new video games released in 2018, which makes finding good partners and games challenging. We want to design an app with the visualizable community network and multiple search filters, so that steam users can easily find friends and games.

## Data

Data: user profile, game profile, users’ friends list, users’ games list, etc.

Source: Valve, a Steam Web API Provider.

https://developer.valvesoftware.com/wiki/Steam_Web_API

Problem: Only users whose data are set public can be collected, and 100,000 rate-limit per day.

## Method

Methods: Link prediction algorithms such as Jaccard, Adamic / Adar, clustering algorithms such as nearest neighbor and affinity propagation, community detection algorithm such as Louvain.

Existing UI Demo: Steam Database, Steam Gauge, Steam Ladder. 

Existing Python Package: scikit-learn, Tensorflow, theano. 

Yes, we will refer and modify them.

## Related Work



GameRecs: Video Games Group Recommendations 

https://link.springer.com/chapter/10.1007/978-3-030-30278-8_49

https://github.com/Nikkilae/group-game-recommender-test



Generating and Personalizing Bundle Recommendations on Steam 

https://cseweb.ucsd.edu/~jmcauley/pdfs/sigir17.pdf

https://cseweb.ucsd.edu/~jmcauley/



Bundle Generation and Group Recommendation applied to the Steam Video Game Platform 

http://snap.stanford.edu/class/cs224w-2018/reports/CS224W-2018-10.pdf

https://github.com/wxy1224/bundle_recommendation



Archetypal Game Recommender Systems 

http://ceur-ws.org/Vol-1226/paper10.pdf



The Steam Engine: A Recommendation System for Steam Users 

http://brandonlin.com/steam.pdf



Condensing Steam: Distilling the Diversity of Gamer Behavior

https://dl.acm.org/citation.cfm?id=2987489



An analysis of the Steam community network evolution

https://ieeexplore.ieee.org/abstract/document/6377133

## Evaluation

Evaluation: For friend recommendation, we don’t need to remove edges randomly because the dates when users became friends are available. We just need to perform training in one time-window and testing in another. And similar for game recommendation.

Baseline: Basic Link prediction algorithms and nearest-neighbor clustering.

Plot: user’s social networks graph

Performance metrics: Mean Reciprocal Rank, Accuracy Score, Confusion Matrix.

