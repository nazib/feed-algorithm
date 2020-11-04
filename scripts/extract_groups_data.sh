#!/bin/bash

user=${MYSQL_USER}
pass=${MYSQL_PASS}
limit=${LIMIT:-5000}
file=${1:-'Data/groups_data.tsv'}

if [[ -z "$user" && -z "$pass" ]]; then
  echo "Please set MYSQL_USER and MYSQL_PASS environment variables"
  exit 1
fi

query=$(cat <<EOF
SELECT
    group_id,
    object_id,
    name,
    type,
    public,
    deactive,
    created_at,
    updated_at
FROM
    travello_live.GROUP
WHERE
    deactive = 0
    AND public = 1;
EOF
)

echo "starting to extract data into file $file"
echo "Query: $query"

mysql --user=$user --password=$pass --host=127.0.0.1 --port=3307 -B -e "$query" > $file

echo "extracted items:" + $(wc -l $file)
echo "done"
echo
