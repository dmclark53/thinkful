-- (Challenge) What's the length of the longest trip for each day it rains anywhere?
WITH 
	rainy
AS (
	SELECT 
		DATE(date) rain_date
	FROM 
		weather
	WHERE 
		events = 'Rain'
	GROUP BY 1
) 

SELECT
	MAX(trips.duration) longest_trip,
	rainy.rain_date
FROM
	rainy
JOIN
	trips
ON
	rainy.rain_date = DATE(trips.start_date)
GROUP BY
	rainy.rain_date
ORDER BY
	longest_trip
DESC;
