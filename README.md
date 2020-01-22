# Steam Explorer: Games Recommendation for Steam Users

This is a project for IIT CS579. See [Instructions.md](Instructions.md) to get started.

# Introduction
A huge number of games are published each year in Steam, which is the largest digital distribution platform for PC gaming. With the progressively expanding number of accessible computer games, Steam users are overwhelmed with searching for new games to play. To solve this problem, we aim to develop a games recommendation framework employing the content-based model and collaborative filtering models. The prediction is based on the number of hours Steam users spend on games in 2014. We find that the collaborative filtering models outperform the content-based model. And inside of collaborative filtering model, the item-based model outperforms the user-based model by providing the highest recall rates and lowest RMSE. In sum, the content-based model is useful to recommend games to players with unique taste and collaborative filtering models are more suitable to do general recommendations. 

# Data
- Data collection
    1. Historical Data provided by [BYU Internet Research Lab](https://steam.internet.byu.edu/)
    2. Real-time data collected by Valve
- Data pre-process
    1. Subset selection
       Select User-game-pair with games count between 20-200,deleted games playtime equals 0. Deleted game feature that is not released or not English.
    2. Text pre-processing
       Remove stop words, lemmatize vocabularies, remove low/high frequency words, transform the descrip-tion of games to tf-idf matrix for building game’sprofile.
    3. Features pre-processing
       Change user-game pair to user-game rating by substituting z-scores for raw playtime. Scaling and normalization features
    4. Split Data
       User game pairs are split into training data and testingdata for K-fold cross-validation where k=5. 
    
# Methods
- Content-based Model
    1. Input:
        - Users-games-matrix
        - Games-features-matrix
    2. The users-features matrix calculated by averaging user owned games feature vectors.
    3. By calculating the cosine-similarity between users' and games' feature vector, we can rank the those games which is not owned yet by a user and give recommendations accordingly.

- User-based Model
    1. Find the K-nearest neighbors (KNN) to the usera, using a similarity function to measure the distance between each pair of users, and in this project, Cosine Similarity will be chosen for determining user’s similarity.
    2. Predict the rating that a user will give to all items the neighbors have consumed but this user has not. We look for a item with the best predictedrating. In other words, we are creating a user-game Matrix, predicting the ratings on items (the active user has not see), based on the other similar users. We alreay get the similarity between users, and the users’ rating for games we already known.For each neighbor in neighbors, we can predict the ratings.

- Item-based Model
    1. Also use Cosine Similarity to calculate similarity among the items. The difference between UB-CF and this model is that, in this case, we directly pre-calculate the similarity between the co-rated items, skipping K neighborhood search.
    2. Second step is predicting rating. Suppose we have a user. Also, we have two items. And user rated one of it, then this user’s rating of another item will be predicted.

# Results
From the recall rate perspective, two collaborative filtering models outperforms content-based model. Users' taste  diversity and popular games concentration make game recommendation using item-item collaborative filtering easier. But the population bias of item-item collaborative filtering model also means that it cannot recommend items to someone with unique taste. Combining these models into a hybrid model can leverage both of their merits, which deserves further investigation.

Choosing which model to adopt does only related to the performance but also computationally speed and size limitation. Content-based model is relatively fast and easy to compute because comparing between users is not needed. Also, considering the fast emerging of new games, CF models are not suitable to predict without the ratings available. 

# WorkFlow
![Workflow](https://drive.google.com/uc?export=view&id=1kMGD6h1bcJLPf4K1brPo0tz3tFHnIJhu)

# How to use
1. First, ensure you have done in the Instructions.md.
2. In your terminal, use this command: osna collect.
3. Wait for the collection finishing, and then use: osna preprocess.
4. Optional: if you want to see the statistics of collected data, after preprocessing step, use this command: osna stats.
5. Wait for the preprocessing finishing, and then use: osna train. In this step, the process will take a long time and consumes a lot of memory in matrix calculation. (about 0.5 hours in a 128GB RAM machine)
6. Optional: if you want to see the evaluation comparison of these three models, after the training finshed, use this command: osna evaluate.
7. Wait for the training finishing, use this command: osna web, then you can open your browser and write URL: localhost:9999, and you will use our application for steam games recommend system.
    - Please note that a 'HOME/.osna/credentials.json' file with following format is needed to make this web working.
    {"steam_api_key": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"}
    [Apply for Steam API key](https://steamcommunity.com/dev/apikey)
8. In the application, first input the userId, and submit it. There are three functions you can access:
    - User Profile: see all informations of the user which you just input
    - Social Network: get the friendship network graph of the user which you just input
    - Games Recommendation: our system will recommend top 30 games for user which you just input


