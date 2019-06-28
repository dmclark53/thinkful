-- Which station is full most often?
SELECT
	status.station_id,
	stations.name,
	-- Counting casing when there is an empty doc
	COUNT(CASE WHEN docks_available=0 THEN 1 END) empty_count
FROM 
	status
JOIN 
	stations
ON
	stations.station_id = status.station_id
GROUP BY 1,2
ORDER BY empty_count DESC;