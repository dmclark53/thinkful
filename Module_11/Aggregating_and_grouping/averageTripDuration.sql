-- What is the average trip duration, by end station?
SELECT
	end_station,
	AVG(duration) AS average_duration
FROM
	trips
GROUP BY end_station
ORDER BY average_duration;