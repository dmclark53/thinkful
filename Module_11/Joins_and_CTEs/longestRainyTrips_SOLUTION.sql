-- What are the three longest trips on rainy days?

-- Rainy days are grouped and placed in the CTE rainy
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
	t.trip_id,
	t.duration,
	DATE(t.start_date)
FROM 
	trips t
JOIN 
	rainy r
ON 
	DATE(t.start_date) = r.rain_date
-- No need to search for maximum duration. Just use ORDER BY.
ORDER BY 
	duration 
DESC
LIMIT 3;