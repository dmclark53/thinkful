-- What are the three longest trips on rainy days?
SELECT
	trips.trip_id,
	MAX(trips.duration) AS max_duration
FROM
	trips
JOIN
	weather
ON
	trips.zip_code = weather.zip
-- This won't work because sometimes it rained multiple times in one day. These
-- days need to be grouped first.
WHERE weather.events = 'Rain'
GROUP BY
	trips.trip_id
ORDER BY
	max_duration
DESC
LIMIT 3;
