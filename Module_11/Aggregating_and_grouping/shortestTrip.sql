-- What's the shortest trip that happened?
SELECT
	trip_id,
	MIN(duration) AS min_duration
FROM
	trips
GROUP BY trip_id
ORDER BY min_duration
LIMIT 1;
