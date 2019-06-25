-- What was the hottest day in our data set? Where was that?
SELECT
	zip,
	date,
	MAX(maxtemperaturef) AS max_temp
FROM
	weather
GROUP BY zip, date
ORDER BY max_temp DESC
LIMIT 1;