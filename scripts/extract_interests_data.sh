#!/bin/bash

user=${MYSQL_USER}
pass=${MYSQL_PASS}
limit=${LIMIT:-5000}
file=${1:-'Data/interests.tsv'}

if [[ -z "$user" && -z "$pass" ]]; then
  echo "Please set MYSQL_USER and MYSQL_PASS environment variables"
  exit 1
fi

query=$(cat <<EOF
SELECT
    id,
    object_id,
    name,
    display_order
FROM
    travello_live.interests
ORDER BY
    display_order ASC;
EOF
)

echo "starting to extract data into file $file"
echo "Query: $query"

mysql --user=$user --password=$pass --host=127.0.0.1 --port=3307 -B -e "$query" > $file

echo "extracted items:" + $(wc -l $file)
echo "done"
echo
