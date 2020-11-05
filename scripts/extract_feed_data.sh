#!/bin/bash

user=${MYSQL_USER}
pass=${MYSQL_PASS}
limit=${LIMIT:-5000}
file=${1:-'Data/feed_data.tsv'}

if [[ -z "$user" && -z "$pass" ]]; then
  echo "Please set MYSQL_USER and MYSQL_PASS environment variables"
  exit 1
fi

query=$(cat <<EOF
SELECT
    f.feed_id,
    f.posted_by AS postUserId,
    f.object_id AS feedObjectId,
    f.likes,
    f.comments,
    f.posted_date,
    Now() as now,
    Timestampdiff(minute, f.posted_date, Now()) as postAge,
    f.groupname,
    f.posted_location,
    f.location,
    Coalesce(fm.numberofmediaurls, 0) AS numberOfMediaUrls,
    Concat('\"', f.text, '\"') as feedText
FROM
    travello_feed_live.feed f
    JOIN -- randomly select ${limit} feed_ids --
        (
          SELECT
              feed_id
          FROM
              travello_feed_live.feed
          ORDER BY
              Rand() LIMIT ${limit}
        )
        AS f2
        ON f.feed_id = f2.feed_id
    LEFT JOIN -- select numberOfMediaUrls from feed_media --
        (
          SELECT
              feed_id,
              Count(*) AS numberOfMediaUrls
          FROM
              travello_feed_live.feed_media
          GROUP BY
              feed_id
        )
        fm
        ON f.feed_id = fm.feed_id
ORDER BY
    f.feed_id ASC
EOF
)

echo "starting to extract data into file $file"
echo "Query: $query"

mysql --user=$user --password=$pass --host=127.0.0.1 --port=3307 -B -e "$query" > $file

echo "extracted items:" + $(wc -l $file)
echo "done"
echo
