-- (Challenge) What's the length of the longest trip for each day it rains anywhere?
-- NOTE: The solution uses two CTEs.

-- First, find all rainy days.
WITH 
	rainy 
AS (
	SELECT 
		DATE(date) weather_date
	From 
		weather
	WHERE 
		events = 'Rain'
	GROUP BY 
		1
),
-- Second, find all rainy trips.
	rain_trips 
AS (
	SELECT
		trip_id,
		duration,
		DATE(trips.start_date) rain_trips_date
	FROM 
		trips
	JOIN 
		rainy
	ON 
		rainy.weather_date = DATE(trips.start_date)
	ORDER BY 
		duration 
	DESC
)
-- Perform agregation at the end.
SELECT 
	rain_trips_date,
	MAX(duration) max_duration
FROM 
	rain_trips
GROUP BY 
	1
ORDER BY 
	max_duration 
DESC