-- Which station is full most often?
SELECT
	stations.name,
	-- Here I am only looking at the average availability. I need to look at the
	-- total number of empty docks.
	AVG(status.bikes_available) AS full_station
FROM
	stations
JOIN
	status
ON
	stations.station_id = status.station_id
GROUP BY
	stations.name
ORDER BY
	full_station
LIMIT 1;