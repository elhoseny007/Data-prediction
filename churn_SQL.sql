--top 10 as total call bills with total call mins
SELECT TOP 10 round(total_bills,3),round(total_mins,3) 
FROM Telecom
ORDER BY total_bills,total_mins DESC;
--customer_compliment which lead to churn out 
SELECT CustServ_Calls,CAST(Churn AS int) AS churn
FROM Telecom
WHERE churn=1 AND CustServ_Calls>0 
-- total customer_compliment which lead to churn out  
SELECT COUNT(*) total_churn
FROM Telecom
WHERE CustServ_Calls>0 AND CAST(churn AS int)=1; 
-- states which have highest 10 compliments 
SELECT TOP 10 State,CustServ_Calls
FROM Telecom
WHERE CustServ_Calls>0
ORDER BY CustServ_Calls DESC
