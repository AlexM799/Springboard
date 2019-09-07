/* SQL mini project. 
Alexia Marcous
Cohort August 2019 */


/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */

/*A1: This solution assumes that by 'members', fees to guests are
not to be included. */

SELECT name 
FROM Facilities 
WHERE membercost > 0


/* Q2: How many facilities do not charge a fee to members? */

/*A2: This solution assumes that by 'members', free bookings for guests 
are excluded. */

SELECT count(name)
FROM Facilities 
WHERE membercost = 0


/* Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

SELECT facid, name, membercost, monthlymaintenance
FROM Facilities
WHERE (membercost / monthlymaintenance) < 0.2


/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */

SELECT *
FROM Facilities
WHERE facid in (1, 5)


/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */

SELECT name, monthlymaintenance,
CASE WHEN monthlymaintenance > 100 THEN 'expensive'
ELSE 'cheap' END AS cheap_or_exp
FROM Facilities


/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */

SELECT  firstname, surname
FROM Members 
WHERE surname <> 'GUEST'
AND joindate =  (SELECT max(joindate) FROM Members)


/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

SELECT DISTINCT CONCAT( name,  ' ', firstname,  ' ', surname ) AS tennis_player
FROM Members
JOIN Bookings ON Members.memid = Bookings.memid
JOIN Facilities ON Bookings.facid = Facilities.facid
WHERE name LIKE  '%Tennis Court%'
AND surname <>  'GUEST'
ORDER BY surname, firstname


/* Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

SELECT name, concat(firstname, ' ', surname) as Member_or_Guest, 
membercost * slots as cost
FROM Members 
JOIN Bookings ON Members.memid = Bookings.memid
JOIN Facilities ON Bookings.facid = Facilities.facid
WHERE date(starttime) = '2012-09-14'
AND Members.memid <> 0
AND membercost * slots > 30
UNION ALL
SELECT name, concat(firstname, ' ', surname) as Member_or_Guest, 
guestcost * slots as cost
FROM Members 
JOIN Bookings ON Members.memid = Bookings.memid
JOIN Facilities ON Bookings.facid = Facilities.facid
WHERE date(starttime) = '2012-09-14'
AND Members.memid = 0
AND guestcost * slots > 30
ORDER BY cost DESC


/* Q9: This time, produce the same result as in Q8, but using a subquery. */

SELECT * FROM
(SELECT name as FacilityName, 
CASE When Members.memid <> 0 Then concat(firstname, ' ', surname) Else 'Guest' End as Member_or_Guest,
CASE When Bookings.memid = 0 Then guestcost * slots Else membercost * slots End
    as booking_cost
FROM Members 
JOIN Bookings ON Members.memid = Bookings.memid
JOIN Facilities ON Bookings.facid = Facilities.facid
WHERE date(starttime) = '2012-09-14'
) as results
WHERE booking_cost > 30
ORDER BY booking_cost DESC


/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

SELECT name, 
SUM(CASE When memid = 0 Then guestcost * slots Else 0 End) +
SUM(CASE When memid <> 0 Then membercost * slots Else 0 End) as Total_Revenue
FROM Bookings
JOIN Facilities ON Bookings.facid = Facilities.facid
GROUP BY Facilities.facid
HAVING Total_Revenue < 1000
ORDER BY Total_Revenue
