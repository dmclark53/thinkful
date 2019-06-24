SELECT
	mintemperaturef AS min_temp
FROM
	weather
WHERE
	events = 'Rain' AND
	zip = 94301;