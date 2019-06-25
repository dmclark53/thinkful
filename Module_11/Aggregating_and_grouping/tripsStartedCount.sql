-- How many trips started at each station?
SELECT
	start_station,
	COUNT(*) AS trip_count
FROM
	trips
GROUP BY start_station
ORDER BY trip_count DESC;
