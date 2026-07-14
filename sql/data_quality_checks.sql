-- Each query must return zero rows for a passing critical check.
-- check: duplicate_loan_id
SELECT loan_id, COUNT(*) AS n FROM loans GROUP BY loan_id HAVING COUNT(*) > 1;

-- check: missing_critical
SELECT loan_id FROM loans
WHERE loan_id IS NULL OR origination_date IS NULL
   OR performance_window_end IS NULL OR default_12m IS NULL;

-- check: missing_model_input
SELECT loan_id FROM loans
WHERE fico_score IS NULL OR ltv IS NULL OR dti IS NULL
   OR loan_amount IS NULL OR interest_rate IS NULL OR rate_spread IS NULL
   OR loan_purpose IS NULL OR occupancy_status IS NULL
   OR prior_delinquency IS NULL OR state IS NULL
   OR unemployment_rate IS NULL OR hpi_yoy_change IS NULL;

-- check: invalid_target
SELECT loan_id, default_12m FROM loans WHERE default_12m NOT IN (0, 1);

-- check: invalid_ranges
SELECT loan_id FROM loans
WHERE fico_score NOT BETWEEN 300 AND 850
   OR ltv NOT BETWEEN 20 AND 150
   OR dti NOT BETWEEN 0 AND 80
   OR loan_amount NOT BETWEEN 25000 AND 1500000
   OR interest_rate NOT BETWEEN 0 AND 25
   OR rate_spread NOT BETWEEN -1 AND 10
   OR prior_delinquency NOT IN (0, 1)
   OR unemployment_rate NOT BETWEEN 2 AND 15
   OR hpi_yoy_change NOT BETWEEN -20 AND 25;

-- check: invalid_categories
SELECT loan_id FROM loans
WHERE split NOT IN ('train','validation','oot')
   OR loan_purpose NOT IN ('purchase','rate_term_refi','cash_out_refi')
   OR occupancy_status NOT IN ('owner','second_home','investor')
   OR state NOT IN ('IL','WI','OTHER');

-- check: incomplete_outcome_window
SELECT loan_id FROM loans
WHERE date(performance_window_end) NOT IN (
        date(origination_date, '+12 months'),
        date(origination_date, '+12 months', '-1 day')
      )
   OR date(performance_window_end) > date('2023-12-31');
