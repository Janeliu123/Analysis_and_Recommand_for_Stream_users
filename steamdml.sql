--Summary

--Print size
SELECT table_name AS "Table",
ROUND(((data_length + index_length) / 1024 / 1024), 2) AS "Size (MB)"
FROM information_schema.TABLES
WHERE table_schema = 'test'
ORDER BY (data_length + index_length) DESC;
--Print rows
SELECT table_name, table_rows
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'steam';


--Games_Daily

--View number of timestamps (8GB RAM, 37.00 sec)
create table games_daily_timestamp as select year(dateretrieved), month(dateretrieved), count(*) from Games_Daily group by year(dateretrieved), month(dateretrieved);
--View number of games owned by each user (8GB RAM, 19.81 sec)
create table games_daily_count_games as select steamid, count(distinct appid) as count_games from Games_Daily group by steamid;
-- (4466 distinct games)
select count(distinct appid) from Games_Daily;
-- (178454 distinct users)
select count(*) from games_daily_count_games;
-- Export distinct users
select steamid from games_daily_count_games
INTO OUTFILE '/var/lib/mysql-files/games_daily_count_games_users.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';
-- Export number of games distribution
select count_games, count(*) from games_daily_count_games group by count_games
INTO OUTFILE '/var/lib/mysql-files/games_daily_count_games_distri.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';
-- Load live data
create table games_daily_count_games_users_distinct (
  `steamid` bigint(20) unsigned NOT NULL,
  `appid` int(10) unsigned NOT NULL,
  `playtime_forever` int(10) unsigned DEFAULT NULL,
  PRIMARY KEY (`steamid`,`appid`)
);
--(1151243 total)
LOAD DATA INFILE '/var/lib/mysql-files/games_daily_count_games_users_valid.csv'
INTO TABLE games_daily_count_games_users_distinct
COLUMNS TERMINATED BY ','
LINES TERMINATED BY '\n';
--(5657 distinct users)
create table games_daily_count_users_distinct as
select distinct steamid from games_daily_count_games_users_distinct;
select count(*) from games_daily_count_users_distinct;
-- Split Games_Daily to train set and test set
-- test dataset (20 min 12.44 sec)
create table games_daily_test as
select distinct steamid, appid, playtime_forever from Games_Daily
natural join games_daily_count_users_distinct;
create table games_daily_test_distinct as
select steamid, appid, max(playtime_forever) as playtime_forever
from games_daily_test
group by steamid, appid;
-- (435466 total, 5657 distinct users)
select count(distinct steamid) from games_daily_test_distinct;
select * from games_daily_test_distinct
INTO OUTFILE '/var/lib/mysql-files/games_daily_test_distinct.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';
-- count new games distribution
create table games_daily_test_new as
select steamid, appid from games_daily_count_games_users_distinct
where (steamid, appid) not in
(select steamid, appid from games_daily_test_distinct);
-- for statistics
create table games_count_stats as
select * from (
(select steamid, count(*) as games_count_history from games_daily_test_distinct group by steamid) as aa
natural join
(select steamid, count(*) as games_count_live from games_daily_count_games_users_distinct group by steamid) as bb
);
create table games_count_stats2 as(
    select games_count_stats.*, aa.games_count_new from games_count_stats left outer join 
    (select steamid, count(*) as games_count_new from games_daily_test_new group by steamid) as aa
    on games_count_stats.steamid = aa.steamid
);
select * from games_count_stats2
INTO OUTFILE '/var/lib/mysql-files/games_count_stats.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';

-- (3480399 total, 172797 distinct users, 14 min 0.99 sec)
-- train dataset
create table games_daily_train as
select steamid, appid, max(playtime_forever) as playtime_forever from Games_Daily
where steamid not in (select steamid from games_daily_test_distinct)
group by steamid, appid;
alter table games_daily_train add primary key (steamid, appid);

-- Select game that has play record (1765219 records)
create table games_daily_train_played as
select * from games_daily_train where playtime_forever > 0;
alter table games_daily_train_played add primary key (steamid, appid);

-- (172562 distinct users)
create table games_daily_train_count as
select steamid, count(*) as count_games 
from games_daily_train_played
group by steamid;

-- (44383 distinct users)
-- create table games_filter as (select steamid from games_daily_train_count where count_games between 20 and 200);
-- alter table games_filter add primary key (steamid);

create table games_filter as (select steamid from games_daily_train_count where count_games between 10 and 200);
alter table games_filter add primary key (steamid);
-- (1319053 records, 29.03 sec)
create table games_daily_train_selected_played as
select * from games_daily_train_played where steamid in (select * from games_filter);

select * from games_daily_train_selected_played
INTO OUTFILE '/var/lib/mysql-files/games_daily_train_selected_played.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';

-- Select game popularity (4309 games, 32.41 sec)
create table games_popularity as
select appid, count(*) as popularity from Games_Daily where playtime_forever > 0 group by appid;

select * from games_popularity
INTO OUTFILE '/var/lib/mysql-files/games_popularity.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';

--Friends

--View number of timestamps (128GB RAM, 5 min 2.58 sec)
create table friends_timestamp as select year(dateretrieved), month(dateretrieved), count(*) from Friends group by year(dateretrieved), month(dateretrieved);
--View number of friends for each user (128GB RAM, 43 min 38.81 sec)
create table friends_count as select steamid_a as steamid, count(distinct steamid_b) as count_friends from Friends group by steamid;
-- (33119781 distince user)
select count(*) from friends_count;
-- sudo mysql --defaults-file=/etc/mysql/debian.cnf
-- Export friends_count (128GB RAM, 33.13 sec)
select * from friends_count 
INTO OUTFILE '/var/lib/mysql-files/friends_count.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';
-- Export number of groups distribution (128GB RAM, 38.19 sec)
select count_friends, count(*) from friends_count group by count_friends
INTO OUTFILE '/var/lib/mysql-files/friends_count_distri.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';
-- Filter Friends within Games_Daily (8GB RAM, 30.30 sec, 98691 users)
create table friends_games_count as 
select * from friends_count natural join games_daily_count_games;
-- Sample 22050 users
select steamid from friends_games_count where count_friends > 10 and count_games > 10 and count_friends < 100
INTO OUTFILE '/var/lib/mysql-files/friends_games_count.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';
-- Explore users friends 
create table friends_filter as
select steamid as steamid_a from friends_games_count where count_friends > 10 and count_games > 10 and count_friends < 100;
-- (659903 distinct id)
create table friends_friends as
(select steamid_b from Friends natural join friends_filter) union friends_filter;
-- Export selected user and their friends' friends pair (128GB RAM, 6 min 30.96 sec, 34726061 rows)
create table friends_selected as
select steamid_a, steamid_b, friend_since from Friends natural join friends_friends;
select * from friends_selected
INTO OUTFILE '/var/lib/mysql-files/friends_selected.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';


--Groups

--View number of timestamps (8GB RAM, 27.04 sec)
create table groups_timestamp as select year(dateretrieved), month(dateretrieved), count(*) from Groups group by year(dateretrieved), month(dateretrieved);
--View number of group for each user (8GB RAM, 16 min 29.01 sec)
create table groups_count as select steamid, count(distinct groupid) as count_groups from Groups group by steamid;
-- (13380609 distince user)
select count(*) from groups_count;
-- Export number of groups distribution
select count_groups, count(*) from groups_count group by count_groups
INTO OUTFILE '/var/lib/mysql-files/groups_count_distri.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';


-- Games info

-- (6512 distince Developer)
select count(distinct Developer) from Games_Developers;

-- (4342 distince Publisher)
select count(distinct Publisher) from Games_Publishers;

-- (22 distince Genre)
select count(distinct Genre) from Games_Genres;

-- (22 distince Apps)
select count(distinct appid) from App_ID_Info;

-- Export Games info
select * from Games_Developers
INTO OUTFILE '/var/lib/mysql-files/Games_Developers.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';
select * from Games_Publishers
INTO OUTFILE '/var/lib/mysql-files/Games_Publishers.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';
select * from Games_Genres
INTO OUTFILE '/var/lib/mysql-files/Games_Genres.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';
select * from App_ID_Info
INTO OUTFILE '/var/lib/mysql-files/App_ID_Info.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n';

