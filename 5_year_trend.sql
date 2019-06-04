select * from posts where owneruserid in 
(select id from users where creationdate >= '20130101'and creationdate < '20130201'
and lastaccessdate >= '20190101'
and reputation < 551
)
and posttypeid = 1
and creationdate >= '20180101'and creationdate < '20190101'
order by owneruserid, creationdate
