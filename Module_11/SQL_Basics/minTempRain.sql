-- The min temperatures of all the occurrences of rain in zip 94301.
SELECT
	mintemperaturef AS min_temp
FROM
	weather
WHERE
	events = 'Rain' AND
	zip = 94301;
