-- Return a list of stations with a count of number of trips starting at that 
-- station but ordered by dock count.

-- I was WAY over thinking this one. I only needed to the trips and stations
-- tables.

-- Do aggregation first
WITH
	trip_count
AS (
	SELECT
		trips.start_station,
		COUNT(trips.trip_id)
	FROM
		trips
	GROUP BY
		trips.start_station
)

SELECT
	trip_count.start_station,
	stations.dockcount,
	status.docks_available,
	COUNT(*)
FROM
	trip_count
JOIN
	stations
ON
	trip_count.start_station = stations.name
JOIN
	status
ON
	stations.station_id = status.station_id
GROUP BY
	1, 2, 3
ORDER BY
	docks_available;
